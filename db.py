#!/usr/bin/env python

import hashlib
import heapq
import itertools
import os
from collections import defaultdict

import cbor2
import sys
import json


SLICE_EXTENSION = '.slices'


def md5(b):
    m = hashlib.md5()
    m.update(b)
    return m.digest()[:6]


def btoh(b):
    """ Convert bytes to hex """
    return ''.join(format(x, f'02x') for x in b)


class ID:
    def __init__(self):
        self.values = {}
        self.max_id = 1

    def get(self, key):
        if key in self.values:
            return self.values[key]
        current_id = self.max_id
        self.max_id += 1
        self.values[key] = current_id
        return current_id

    def getIDList(self):
        return [
            dict(
                ID = id,
                value = value
            ) for value, id in self.values.items()
        ]

def load_cbor_file(path, include_detail=False):
    file_ids = ID()
    func_ids = ID()
    entries = []

    with open(path, 'rb') as fd:
        try:
            while True:
                line = cbor2.load(fd)
                text = line['canonicalText']
                slice_hash = md5(text.encode('ascii'))

                if include_detail:
                    entry = (slice_hash, line)
                else:
                    path = line['file']['path']
                    func_address = line['function']['address']

                    file_id = file_ids.get(path)
                    func_id = func_ids.get(func_address)

                    entry = (slice_hash, file_id, func_id)

                entries.append(entry)
        except cbor2.CBORDecodeEOF:
            pass

    header = dict(
        files=[dict(id=fid, path=path) for path, fid in file_ids.values.items()]
    )

    return header, sorted(entries, key=lambda x:x[0])


def convert_to_counts(entries):
    entry_groups = []
    for k, g in itertools.groupby(entries, lambda x: x[0]):
        entry_groups.append(list(g))

    counts = []
    for group in entry_groups:
        slice_hash = group[0][0]

        funcs = set()
        instance_counts = len(group)
        func_counts = len(set(func_id for h, fid, func_id in group))
        fids = set(fid for h, fid, func_id in group)
        file_counts = len(fids)
        file_list = sorted(fids)

        count_entry = [slice_hash, file_counts, func_counts, instance_counts, sorted(fids)]
        counts.append(count_entry)

    return counts


def merge(iterators):
    iters = [iter(it) for it in iterators]
    items = []
    for i, it in enumerate(iters):
        try:
            items.append((next(it), i))
        except StopIteration:
            pass

    heapq.heapify(items)

    while items:
        value, index = heapq.heappop(items)

        # Produce the index as well for later lookups (e.g. when doing ID number thunks)
        yield value, index

        try:
            heapq.heappush(items, (next(iters[index]), index))
        except StopIteration:
            pass

class DBMain:
    def __init__(self, args):
        self.all_counts = []
        self.headers = []
        self.db_header = {}

        self.args = args

        if self.args.search:
            search_results = self.search()
            if len(self.args.files) > 1:
                self.generateMatchSetCorrelationMatrix(search_results)
        else:
            self.ingest_files()

    def base_file_name(self, path):
        return os.path.basename(path).replace(SLICE_EXTENSION, '')


    def generateMatchSetCorrelationMatrix(self, match_results):
        ''' This may look like a confusion matrix, but it isn't.  It counts the number of match sets that only
            contain files in the search list.
        '''

        search_filenames = self.getSearchFilenames()

        confusion_matrix = {}

        for filename, match_results in match_results.items():
            row = defaultdict(int)
            for result in match_results:
                fileNames = result['otherFiles']['fileNames']
                good_match = all(fname in search_filenames for fname in fileNames)

                if good_match:
                    for fname in fileNames:
                        row[fname] += 1

            confusion_matrix[filename] = row

        matrix_order = confusion_matrix.keys()

        for row_index in matrix_order:
            row = confusion_matrix[row_index]
            row_txt = ' '.join(f'{row.get(col_index,""):4}' for col_index in matrix_order)
            print(f'{row_index:16} {row_txt}')

        print(f',{",".join(matrix_order)}')
        for row_index in matrix_order:
            row = confusion_matrix[row_index]
            row_txt = ','.join(str(row.get(col_index, 0)) for col_index in matrix_order)
            print(f'{row_index},{row_txt}')

    def getSearchFilenames(self):
        return [self.base_file_name(path) for path in self.args.files]

    def search(self):
        search_filenames = self.getSearchFilenames()

        all_results = {}

        last_n = 100 if len(self.args.files) == 1 else 10

        for path in self.args.files:
            filename = self.base_file_name(path)

            match_results = self.search_file(path)
            # match_results = sorted(match_results, key=lambda result:result['thisFile']['funcNames'])

            match_results = sorted(match_results,
                                   key=lambda result:
                                   (result['otherFiles']['fileCount'],
                                    result['otherFiles']['fileNames'],
                                    )
                                   )

            #filter_match_results = [result for result in match_results if len(result['canonicalText'].split('\n')) > 5]
            # self.display_search_results(filter_match_results)

            match_set_groups = self.group_match_sets(match_results)

            print()
            self.summarize_match_sets(match_set_groups, last_n)

            if self.args.detail:
                self.output_match_result_detail(self.get_detail_file_name(), match_results)

                self.output_match_set_groups(self.args.detail, match_set_groups)

            all_results[filename] = match_results
        return all_results

    def get_detail_file_name(self):
        return self.args.detail.rstrip('/\\') + '.detail'

    def read_db_header(self):
        with open(self.args.db, 'rb') as fd:
            self.db_header = cbor2.load(fd)

    def read_db(self):
        with open(self.args.db, 'rb') as fd:
            try:
                self.db_header = cbor2.load(fd)
                while True:
                    yield cbor2.load(fd)
            except cbor2.CBORDecodeEOF:
                pass

    def group_match_sets(self, match_results) -> list:
        groups = []
        for matchSetHash, group in itertools.groupby(match_results, key=lambda x:x['otherFiles']['matchSetHash']):
            group = list(group)
            otherFiles = group[0]['otherFiles']
            fids = otherFiles['fileIDs']
            fileNames = otherFiles['fileNames']
            groups.append((matchSetHash, fids, fileNames, group))
        return groups

    def output_match_result_detail(self, output_file, match_results):
        with open(output_file, 'wb') as fd:
            for result in match_results:
                cbor2.dump(result, fd)

    def output_match_set_groups(self, output_dir, match_set_groups):
        print(f'output_dir is {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        def formatFuncAddress(funcAddress):
            addrs = [f'{addr:x}' for addr in sorted(funcAddress['addressSet'])]
            return f'{funcAddress["funcName"]:20} {",".join(addrs)}'

        for matchSetHash, fids, fileNames, matchResults in match_set_groups:

            names = (' '.join(fileNames))[:50]
            out_path = os.path.join(output_dir, f'{btoh(matchSetHash)} {names}.txt')

            with open(out_path, 'w') as fd:
                print(f'Match set {btoh(matchSetHash)} has {len(fileNames)} files', file=fd)
                for name in fileNames:
                    print(f'  {name}', file=fd)

                print(f'\nSlices with the match set:', file=fd)

                for result in matchResults:
                    slice_hash = result['hash']
                    dfilText = result['canonicalText']

                    thisFile = result['thisFile']
                    funcAddresses = thisFile['funcAddresses']

                    otherFiles = result['otherFiles']
                    fileCount = otherFiles['fileCount']
                    funcCount = otherFiles['funcCount']
                    instanceCount = otherFiles['instanceCount']

                    count_txt = f'{fileCount} {funcCount} {instanceCount}'

                    single_func = formatFuncAddress(funcAddresses[0]) if len(funcAddresses) == 1 else ''

                    print(f'Slice {btoh(slice_hash)} {count_txt:10} {single_func}', file=fd)

                    if len(funcAddresses) > 1:
                        for funcAddress in funcAddresses:
                            print(f'  {formatFuncAddress(funcAddress)}', file=fd)


                for result in matchResults:
                    slice_hash = result['hash']
                    dfilText = result['canonicalText']

                    otherFiles = result['otherFiles']
                    fileCount = otherFiles['fileCount']
                    funcCount = otherFiles['funcCount']
                    instanceCount = otherFiles['instanceCount']

                    count_txt = f'{fileCount} {funcCount} {instanceCount}'
                    single_func = formatFuncAddress(funcAddresses[0]) if len(funcAddresses) == 1 else ''

                    # Indent DFIL text block
                    dfilText = '\n'.join(f'    {line}' for line in dfilText.split('\n'))

                    print(f'\nSlice {btoh(slice_hash)} {count_txt} {single_func}', file=fd)

                    print(dfilText, file=fd)


    def summarize_match_sets(self, match_set_groups, last_n=50):
        # Sort by number of slices, then by the list of file names.

        groups = sorted(match_set_groups, key=lambda x: (len(x[3]), x[2]))
        for matchSetHash, fids, fileNames, matchResults in groups[-last_n:]:
            names = (' '.join(fileNames))[:150].rstrip()
            print(f'{len(matchResults):6} {btoh(matchSetHash)} {len(fids):3} {names}')

            '''
            for result in matchResults[:5]:
                canonicalText = result['canonicalText']

                n_lines = len(canonicalText.split('\n'))
                dfil_summary = canonicalText.replace('DFIL_', '').replace('DECLARE_', '').replace('\n', '  ')
                print(f'     DFIL {n_lines:3} {dfil_summary[:100]}')
            '''

    def display_search_results(self, match_results):
        for result in match_results:
            sliceHash = result['hash']
            thisFile = result['thisFile']
            otherFiles = result['otherFiles']
            canonicalText = result['canonicalText']

            fileCount = otherFiles['fileCount']
            funcCount = otherFiles['funcCount']
            instanceCount = otherFiles['instanceCount']
            fileIDs = otherFiles['fileIDs']
            fileNames = otherFiles['fileNames']

            funcNames = thisFile['funcNames']
            addressSet = thisFile['allAddresses']

            func_name_txt = ",".join(funcNames)[:30]

            count_txt = f'{fileCount:4} {funcCount:6} {instanceCount:6} '
            count_txt += f'{len(funcNames):4} '

            dfil_exprs = canonicalText.split('\n')

            addrSetText = ','.join(f'{addr:x}' for addr in sorted(addressSet))[:30]


            file_name_txt = ' '.join(sorted(fileNames))[:50]

            dfil_summary = '  '.join(dfil_exprs).replace('DFIL_', '').replace('DECLARE_', '')

            print(f'{btoh(sliceHash)} {count_txt} {func_name_txt:30}  ' +
                  f'{len(fileNames):5} {file_name_txt:50}  ' +
                  f'{len(addressSet):3} {addrSetText:30}  ' +
                  f'{len(dfil_exprs):3} {dfil_summary[:100]}')
            #print(ctext)

    def search_file(self, path: str):
        header, hash_data = load_cbor_file(path, include_detail=True)
        self.read_db_header()
        db_stream = self.read_db()
        h, fileCount, funcCount, instCount, fids = next(db_stream)

        fid_lookup = {
            file['id']: file['path'] for file in self.db_header['files']
        }
        print(f'hash_data = {str(hash_data)[:100]}')
        match_results = []
        for slice_hash, group in itertools.groupby(hash_data, key=lambda x:x[0]):
            group = list(group)

            # Advance DB cursor until its hash is not less than the slice hash
            while h < slice_hash:
                h, fileCount, funcCount, instCount, fids = next(db_stream)

            if h != slice_hash:
                continue

            # Combine the list of this file's addresses
            allAddresses = set()
            for _, line in group:
                aset = line['addressSet']
                allAddresses |= set(line['addressSet'])

            funcAddresses = [
                dict(
                    funcName = line['function']['name'],
                    addressSet = sorted(line['addressSet'])
                ) for _, line in group
            ]

            matchSetHash = md5(','.join(str(fid) for fid in fids).encode('ascii'))

            print(f'allAddresses = {allAddresses}')
            result = dict(
                hash=slice_hash,
                canonicalText=group[0][1]['canonicalText'],
                thisFile=dict(
                    funcNames=sorted([line['function']['name'] for _, line in group]),
                    allAddresses=sorted(allAddresses),
                    funcAddresses=funcAddresses,
                ),
                otherFiles=dict(
                    fileCount=fileCount,
                    funcCount=funcCount,
                    instanceCount=instCount,
                    fileIDs=fids,
                    matchSetHash=matchSetHash,
                    fileNames=[os.path.basename(fid_lookup[fid]) for fid in fids],
                )
            )
            match_results.append(result)
        return match_results

    def ingest_files(self) -> None:
        file_count = 0
        for path in self.args.files:
            if os.path.isdir(path):
                file_count += self.process_folder(path)
            else:
                self.process_file(path)
                file_count += 1

        merged_counts = list(merge(self.all_counts))
        print(f'Merged counts {len(merged_counts)}')
        grouped_db = self.thunk_groups(merged_counts)

        items_written = self.write_to_file(self.args.db, self.db_header, grouped_db)
        print(f'Wrote {items_written} items from {file_count} files to {self.args.db}')

    def process_file(self, path: str) -> None:
        header, entries = load_cbor_file(path)
        counts = convert_to_counts(entries)

        self.headers.append(header)
        self.all_counts.append(counts)

        for hash, file_counts, func_counts, instance_counts, fids in counts:
            fids_txt = ','.join(str(fid) for fid in sorted(fids))

            # print(f'{btoh(hash)} {file_counts:3} {func_counts:3} {len(counts):3} {fids_txt}')

        print(f'{len(entries):6} slices, {len(counts):6} unique {path}')

    def process_folder(self, path: str) -> int:
        file_count = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith(SLICE_EXTENSION):
                    continue
                file_path = os.path.join(root, file)
                self.process_file(file_path)
                file_count += 1
        return file_count

    def _thunk_group_gen(self, merged_counts, id_thunks):
        for slice_hash, group in itertools.groupby(merged_counts, lambda x: x[0][0]):
            file_counts = 0
            func_counts = 0
            instance_counts = 0
            file_ids = set()
            for counts, index in group:
                _, fileCount, funcCount, instCount, fids = counts
                file_counts += fileCount
                func_counts += funcCount
                instance_counts += instCount

                thunk = id_thunks[index]
                file_ids |= set(thunk[fid] for fid in fids)

            yield [slice_hash, file_counts, func_counts, instance_counts, sorted(file_ids)]

    def thunk_groups(self, merged_counts):
        new_id = ID()
        id_thunks = []
        for header in self.headers:
            current_thunk = {}
            files = header['files']
            for document in files:
                id = document['id']
                path = document['path']
                current_thunk[id] = new_id.get(path)

            id_thunks.append(current_thunk)

        self.db_header = dict(
            files=[dict(id=fid, path=path) for path, fid in new_id.values.items()]
        )

        return self._thunk_group_gen(merged_counts, id_thunks)

    def write_to_file(self, path, header, items):
        n_written = 0
        with open(path, 'wb') as fd:
            cbor2.dump(header, fd)
            for item in items:
                cbor2.dump(item, fd)
                n_written += 1
        return n_written

