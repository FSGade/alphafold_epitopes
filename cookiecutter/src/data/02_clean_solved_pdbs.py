#!/usr/bin/env python3

import os
import glob

from Bio.PDB import PDBParser, PDBIO, Select

save_chains_list = [x.strip() for x in open("../../data/interim/non_redundant_chain_ids.txt", "r").readlines()]

solved_pdbs_dir = "../../data/raw/solved_pdbs/"
cleaned_solved_pdbs_dir = "../../data/processed/cleaned_solved_pdbs/"

# Main
for pdb_name in save_chains_list:

    pdb_id, chain_name = pdb_name.split("_")
    
    path = solved_pdbs_dir + pdb_id + ".pdb"

    pdb = PDBParser(PERMISSIVE=True).get_structure(pdb_name, path)

    print(f"Cleaning {pdb_id} and extracting chain {chain_name}")

    if not chain_name == "A":
        rename1, rename2 = {"A": "XXX"}, {chain_name: "A"}
    else:
        rename1, rename2 = {},{}

    for model in pdb:
        for chain in model:
            old_name = chain.get_id()
            new_name = rename1.get(old_name)
            if new_name:
                print(f"renaming chain {old_name} to {new_name}")
                chain.id = new_name
        for chain in model:
            old_name = chain.get_id()
            new_name = rename2.get(old_name)
            if new_name:
                print(f"renaming chain {old_name} to {new_name}")
                chain.id = new_name

    class NonHetSelect(Select):
#    def __init__(self, chain_accept):
#        super(NonHetSelect).__init__()
#        self.chain = chain_accept

        def accept_residue(self, residue):
            return 1 if residue.id[0] == " " else 0

        def accept_chain(self, chain):
            return 1 if chain.id == "A" else 0

    io = PDBIO()
    io.set_structure(pdb)
    io.save(cleaned_solved_pdbs_dir+pdb_name+".pdb", NonHetSelect())

