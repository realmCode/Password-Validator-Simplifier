import glob

BUFFER_SIZE = 10*1024 * 1024  # 1 MB chunks, adjust as needed

files = glob.glob("wordlists/*.txt")

with open("merged.txt", "w", encoding="utf-8", errors='ignore') as out_f:   # use "w" to start fresh
    for path in files:
        with open(path, "r", encoding="utf-8", errors='ignore') as in_f:
            while True:
                chunk = in_f.read(BUFFER_SIZE)
                if not chunk:
                    break
                out_f.write(chunk)
