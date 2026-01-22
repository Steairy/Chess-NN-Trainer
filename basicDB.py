class DB:
    def __init__(self, file="db.bin"):
        self.file = file
        self.db = {}

        with open(self.file, "ab+") as f:
            f.seek(0)
            lines = f.readlines()
        
        for line in lines:
            tokens = line.decode("utf-8").strip("\n").split("\x1e")
            self.db[tokens[0]] = tokens[1]
    
    def write(self, key, value):
        self.db[key] = value
    
    def read(self, key):
        return self.db[key]
    
    def delete(self, key):
        self.db.pop(key)
    
    def flush(self):
        with open(self.file, "wb") as f:
            f.writelines([bytes(f"{key}\x1e{self.db[key]}\n", encoding="utf-8") for key in self.db])
