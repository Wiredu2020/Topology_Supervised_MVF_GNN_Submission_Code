# Graph datasets

The benchmark inputs in `src/data/` are the supplied original files. The TU Dortmund graph-classification datasets are available from the public Graph Kernel Dataset Collection at `https://www.chrsmrrs.com/graphkerneldatasets/`.

Available molecular and biological archives: BZR, COX2, DHFR, ENZYMES, PROTEINS, NCI109, NCI1, PTC_MR. BACE is provided as a SMILES CSV file; conversion to graph data requires RDKit.

Graph-structure inputs: CSL and BREC, supplied as the original publicly distributed archives. Consult their respective original benchmark repositories for citation and usage information.

The following datasets appear in the supplementary numerical results but their raw archives were not present in the recovered materials:

- `PTC_FM`: https://www.chrsmrrs.com/graphkerneldatasets/PTC_FM.zip
- `PTC_FR`: https://www.chrsmrrs.com/graphkerneldatasets/PTC_FR.zip
- `PTC_MM`: https://www.chrsmrrs.com/graphkerneldatasets/PTC_MM.zip
- `Mutagenicity`: https://www.chrsmrrs.com/graphkerneldatasets/Mutagenicity.zip

Third-party input data retain the rights and citation requirements of their respective providers.
