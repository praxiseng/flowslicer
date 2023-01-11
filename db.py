import hashlib
import heapq
import itertools
import os

import cbor2
import sys
import json


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

def load_cbor_file(path):
    file_ids = ID()
    func_ids = ID()
    entries = []

    with open(path, 'rb') as fd:
        try:
            while True:
                line = cbor2.load(fd)

                path = line['file']['path']
                func_address = line['function']['address']
                text = line['canonicalText']

                file_id = file_ids.get(path)
                func_id = func_ids.get(func_address)
                slice_hash = md5(text.encode('ascii'))
                entries.append((slice_hash, file_id, func_id))

        except cbor2.CBORDecodeEOF:
            pass

    header = dict(
        files=[dict(id=fid, path=path) for path, fid in file_ids.values.items()]
    )

    return header, sorted(entries)


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

class Main:
    def __init__(self):
        self.all_counts = []
        self.headers = []
        self.merged_header = {}

        for path in sys.argv[1:]:
            if os.path.isdir(path):
                self.process_folder(path)
            else:
                self.process_file(path)

        merged_counts = list(merge(self.all_counts))
        print(f'Merged counts {len(merged_counts)}')
        #for count, index in merged_counts:
        #    slice_hash, file_counts, func_counts, instance_counts, fids = count
        #    print(f'Item: {btoh(slice_hash)} {index} {file_counts:3} {func_counts:3} {fids}')
        grouped_db = self.thunk_groups(merged_counts)

        #for slice_hash, file_counts, func_counts, instance_counts, fids in grouped_db:
        #    print(f'Item: {btoh(slice_hash)} {file_counts:3} {func_counts:3} {fids}')
        out_file = 'slicedb.db'
        items_written = self.write_to_file(out_file, self.merged_header, grouped_db)
        print(f'Wrote {items_written} items to {out_file}')


    def process_file(self, path):
        header, entries = load_cbor_file(path)
        counts = convert_to_counts(entries)

        self.headers.append(header)
        self.all_counts.append(counts)

        for hash, file_counts, func_counts, instance_counts, fids in counts:
            fids_txt = ','.join(str(fid) for fid in sorted(fids))

            # print(f'{btoh(hash)} {file_counts:3} {func_counts:3} {len(counts):3} {fids_txt}')

        print(f'{len(entries):6} slices, {len(counts):6} unique {path}')

    def process_folder(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith('.cbor'):
                    continue
                file_path = os.path.join(root, file)
                self.process_file(file_path)

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

        self.merged_header = dict(
            files=[dict(id=fid, path=path) for path, fid in new_id.values.items()]
        )


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

    def write_to_file(self, path, header, items):
        n_written = 0
        with open(path, 'wb') as fd:
            cbor2.dump(header, fd)
            for item in items:
                cbor2.dump(item, fd)
                n_written += 1
        return n_written




if __name__ == "__main__":
    Main()