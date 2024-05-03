import lzma
import struct


class Instruction:
    def __init__(self, inst_struct=None) -> None:
        if inst_struct is None:
            self.ip = 0
            self.isbranch = 0
            self.branch_taken = 0
            self.destination_regs = []
            self.source_regs = []
            self.destination_memory = []
            self.source_memory = []
        else:
            self.ip = inst_struct[0]
            self.isbranch = inst_struct[1]
            self.branch_taken = inst_struct[2]
            self.destination_regs = [inst_struct[i] for i in range(3, 5)]
            self.source_regs = [inst_struct[i] for i in range(5, 9)]
            self.destination_memory = [inst_struct[i] for i in range(9, 11)]
            self.source_memory = [inst_struct[i] for i in range(11, 15)]

    def __str__(self) -> str:
        return f"""
        Instruction Pointer: {hex(self.ip)}
        Is Branch: {self.isbranch}
        Branch taken: {self.branch_taken}
        Destination regs: {self.destination_regs}
        Source regs: {self.source_regs}
        Destination memory: {[hex(i) for i in self.destination_memory]}
        Source memory: {[hex(i) for i in self.source_memory]}
        """


with lzma.open("429.mcf-22B.champsimtrace.xz", mode="rb") as f:
    for i in range(32):
        read_bytes = f.read(size=64)
        inst_struct = struct.unpack("<QBB2B4B2Q4Q", read_bytes)
        instruction = Instruction(inst_struct=inst_struct)
        print(instruction)
