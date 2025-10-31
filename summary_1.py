import argparse
import json
import re
from collections import defaultdict

def sec_ranges(seconds):
  
    if not seconds:
        return []
    ranges = []
    start = prev = seconds[0]
    for s in seconds[1:]:
        if s == prev + 1:
            prev = s
        else:
            ranges.append((start, prev))
            start = prev = s
    ranges.append((start, prev))
    return ranges

def format_ranges(ranges):
    
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)

def parse_summary(text):

    cars = {}
    persons = {}
    

    m = re.search(r"Cars:\s*(.*?)\.\s", text)
    if m:
        for part in m.group(1).split(','):
            part = part.strip()
            pm = re.match(r"(id\d+)\s+(\w+)", part)
            if pm:
                cars[pm.group(1)] = pm.group(2)


    m = re.search(r"Persons:\s*(.*?)\.\s", text)
    if m:
        for part in m.group(1).split(','):
            part = part.strip()
            pm = re.match(r"(id\d+)\s+color\s+is\s+(\w+)\s+and\s+posture\s+is\s+(\w+)", part)
            if pm:
                persons[pm.group(1)] = {
                    'color': pm.group(2),
                    'posture': pm.group(3)
                }

 
    categories = ["Extremely near", "Near", "Moderately near", "Moderately far", "Far", "Extremely far"]
    distances = {cat: [] for cat in categories}
    for cat in categories:
        pat = rf"{cat}:\s*(.*?)\."
        m2 = re.search(pat, text)
        if m2:
            for pair in m2.group(1).split(';'):
                pair = pair.strip().replace(' and ', ',')
                ids = [p.strip() for p in pair.split(',') if p.strip()]
                if len(ids) == 2:
                    distances[cat].append((ids[0], ids[1]))

    return cars, persons, distances

def main():
    parser = argparse.ArgumentParser(description="Summarize object appearances and distance relations")
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument("output_file", help="Path to output summary text file")
    args = parser.parse_args()

    data = json.load(open(args.input_file, 'r', encoding='utf-8'))

    info = {}
    categories = ["Extremely near", "Near", "Moderately near", "Moderately far", "Far", "Extremely far"]

    for frame in data.get('frames', []):
        fid = frame.get('frame_id')
        sec = fid // 12
        text = frame.get('summary', '')
        cars, persons, dists = parse_summary(text)

    
        for oid, col in cars.items():
            rec = info.setdefault(oid, {
                'type': 'Car',
                'colors': {},
                'postures': {},
                'secs': set(),
                'relations': {cat: defaultdict(set) for cat in categories}
            })
            rec['colors'].setdefault(sec, col)
            rec['secs'].add(sec)

        for oid, attr in persons.items():
            rec = info.setdefault(oid, {
                'type': 'Person',
                'colors': {},
                'postures': {},
                'secs': set(),
                'relations': {cat: defaultdict(set) for cat in categories}
            })
            rec['colors'].setdefault(sec, attr['color'])
            rec['postures'].setdefault(sec, attr['posture'])
            rec['secs'].add(sec)


        for cat, pairs in dists.items():
            for a, b in pairs:
                if a in info and b in info:
                    info[a]['relations'][cat][b].add(sec)
                    info[b]['relations'][cat][a].add(sec)


    with open(args.output_file, 'w', encoding='utf-8') as out:
        for oid, rec in sorted(info.items(), key=lambda x: x[0]):
            otype = rec['type']
            secs = sorted(rec['secs'])
            ranges_str = format_ranges(sec_ranges(secs))
            color = next(iter(rec['colors'].values()))
            line = f"{otype} {oid} with color {color}"
            if otype == 'Person':
                posture = next(iter(rec['postures'].values()))
                line += f" and posture {posture}"
            line += f" appears during seconds {ranges_str}.\n"
            out.write(line)

            for cat in categories:
                for other, sset in rec['relations'][cat].items():
                    other_type = info[other]['type']
                    sr = format_ranges(sec_ranges(sorted(sset)))
                    out.write(f"It is \"{cat}\" to {other} ({other_type}) during seconds {sr}.\n")
            out.write("\n")

  

if __name__ == '__main__':
    main()
