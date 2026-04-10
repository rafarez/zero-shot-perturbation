# gene_alias_map.py
# Maps benchmark target_gene entries → list of canonical HGNC symbols
# Sources: NCBI Gene, HGNC, UniProt

GENE_ALIAS_MAP: dict[str, list[str]] = {
    # Protein alias → HGNC symbol
    "BAFF":    ["TNFSF13B"],
    "CD20":    ["MS4A1"],
    "OX40":    ["TNFRSF4"],
    "TL1A":    ["TNFSF15"],

    # Immunoglobulin — Omalizumab targets IgE protein (IGHE gene)
    # Perturbation semantics are weak here; flag in results
    "IgE":     ["IGHE"],

    # PDE4 family — perturb all 4 isoforms
    "PDE4":    ["PDE4A", "PDE4B", "PDE4C", "PDE4D"],

    # microRNA — absent from standard mRNA-seq; no valid perturbation target
    # Obefazimod's mechanism is via MIR124-1 upregulation (agonist, not knockdown)
    # Flag this pair as unresolvable in expression space
    "MIR124-1": [],   # empty → pipeline will log warning and skip
}


# Create context manager to save list of target genes
# Not found during the pertubation pipeline.
# For each target gene not found, we save the drug affected by it, 
# the disease this drug is trying to address
import json
class MissingTargetGenesList:
    def __enter__(self):
        return self
    
    def __init__(self, out=None):
        self._missing = {}
        self._output_fn = out
        if out is None:
            self._output_fn = "missing_target_genes.json"
        

    def update(self, gene, disease, drug):
        if gene not in self._missing.keys():
            self._missing[gene] = set()
        
        if (disease, drug) not in self._missing[gene]:
            self._missing[gene].add( (disease, drug) )

    def __exit__(self, exc_type, exc_val, exc_tb):
        json_dict = {}
        for k,v in self._missing.items():
            json_dict[k] = list(v)
        with open(self._output_fn, 'w') as f:
            json.dump(json_dict, f)