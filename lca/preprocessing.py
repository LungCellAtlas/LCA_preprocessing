import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import collections
import glob
from scipy import sparse
from utils import check_if_nan

def get_gene_renamer_dict(gene_names):
    """This function detects genes that are misnamed. Note that it should only
    be run on adata.var.index after removing duplicate gene names! 
    It detects 1) genes with double dashes ("--") and translates them 
    to double underscores ("__"); 2) genes that occur both with a 
    "1" suffix (".1", "-1" or "_1") and without suffix, and translated 
    the with-suffix genes to the without-suffix versions, and 3)
    it detects genes that occur with corresponding but different suffixes 
    (".[suffix]" and "-[suffix]") and translates them to either 1) a version 
    without suffix, if the suffix is ".[single_digit]", or else 2) to the dash-version
    of the suffix, since this is usually the ensembl version. 
    Input: list of gene names
    Returns a dictionary with original gene names as keys, and proposed 
    translations as values."""
    # first check if there are no duplicate gene names:
    if len(gene_names) > len(set(gene_names)):
        print()
        raise ValueError("There are still duplicate gene names in your gene list, \
this function does not work properly with duplicate names. Exiting.")
    # rename genes with double dash to genes with double underscore:
    genes_with_double_dash = [gene for gene in gene_names if "--" in gene]
    genes_to_change_suffix_dict = dict()
    verbose = False
    # find genes that have a suffixed and non-suffixed version
    # (this will detect both genes that only have a non-suffixed version,
    # and genes that have variable suffixes and a non-suffixed version)
    for gene in gene_names:
        # check if it has a suffix:
        if len(gene) > 1:
            if gene[-2:] in [".1", "-1", "_1"]:
                # check if it has a non-suffix version:
                if gene[:-2] in gene_names:
                    genes_to_change_suffix_dict[gene] = gene[:-2]
    # find all genes that have both a .[suffix] and a -[suffix] version
    # (this differs from the loop above, since it will find genes that don't
    # have a short version in the data):
    # a = [x.replace("-", ".") for x in gene_names]
    a = [
        rreplace(
            original_gene_name, old="-", new=".", occurrence=1, only_if_not_followed_by="."
        )
        for original_gene_name in gene_names
    ]
    from_dash_to_dot = [
        item for item, count in collections.Counter(a).items() if count > 1
    ]
    for gene in from_dash_to_dot:
        # get gene name without suffix
        gene_minus_suffix_list = gene.split(".")
        gene_minus_suffix = ".".join(gene_minus_suffix_list[:-1])
        # map "-" version of name to new name
        dot_position = gene.rfind(".")
        gene_as_list = list(gene)
        gene_as_list_dash = gene_as_list
        gene_as_list_dash[dot_position] = "-"
        gene_dash_name = "".join(gene_as_list_dash)
        # if the suffix is a .[single_digit], change it to the name without suffix:
        if gene.split(".")[-1] in [str(digit) for digit in range(0, 10)]:
            # map "." version of name to new name
            genes_to_change_suffix_dict[gene] = gene_minus_suffix
            genes_to_change_suffix_dict[gene_dash_name] = gene_minus_suffix
            # else, change the dot version to the dash-version of the name
            # (this is usually the ensembl version):
        else:
            genes_to_change_suffix_dict[gene] = gene_dash_name
    # now create the remapping dictionary
    genes_remapper = dict()
    for gene in genes_with_double_dash:
        genes_remapper[gene] = gene.replace("--", "__")
    for gene_name_old, gene_name_new in genes_to_change_suffix_dict.items():
        genes_remapper[gene_name_old] = gene_name_new
    return genes_remapper


def add_up_duplicate_gene_name_columns(adata, print_gene_names=True, verbose=False):
    """ This function finds duplicate gene names in adata.var (i.e. duplicates 
    in the index of adata.var). For each cell, it adds up the counts of columns 
    with the same gene name, removes the duplicate columns, and replaces the 
    counts of the remaining column with the sum of the duplicates.
    Returns anndata object."""
    duplicate_boolean = adata.var.index.duplicated()
    duplicate_genes = adata.var.index[duplicate_boolean]
    print("Number of duplicate genes: " + str(len(duplicate_genes)))
    if print_gene_names == True:
        print(duplicate_genes)
    columns_to_replace = list()
    columns_to_remove = list()
    new_columns_array = np.empty((adata.shape[0], 0))
    for gene in duplicate_genes:
        if verbose:
            print("Calculating for gene", gene)
        # get rows in adata.var with indexname equal to gene
        # indexing zero here is to get rid of tuple output and access array
        gene_colnumbers = np.where(adata.var.index == gene)[0]
        # isolate the columns
        gene_counts = adata.X[:, gene_colnumbers]
        # add up gene counts and add new column to new_columns_array
        new_columns_array = np.hstack((new_columns_array, np.sum(gene_counts, axis=1)))
        # store matching column location in real adata in list:
        columns_to_replace.append(gene_colnumbers[0])
        # store remaining column locations in columns to remove:
        columns_to_remove = columns_to_remove + gene_colnumbers[1:].tolist()
    # replace first gene column with new col:
    adata.X[:, columns_to_replace] = new_columns_array
    # remove remaining duplicate columns:
    columns_to_keep = [
        i for i in np.arange(adata.shape[1]) if i not in columns_to_remove
    ]
    adata = adata[:, columns_to_keep].copy()
    if verbose:
        print("Done!")
    return adata
    

def add_cell_annotations(
    adata
    ):
    """ This function adds annotation to anndata:  
    cell level:  
    total_counts, log10_total_counts, n_genes_detected, mito_frac, ribo_frac,   
    compl(exity)  
    gene_level:  
    n_cells  
    Returns anndata object with annotations  
    """
    # cells:
    # total transcript count per cell
    adata.obs['total_counts'] = np.sum(adata.X, axis=1)
    adata.obs['log10_total_counts'] = np.log10(adata.obs['total_counts'])
    # number of genes expressed
    # translate matrix to boolean (True if count is larger than 0):
    boolean_expression = adata.X > 0
    adata.obs['n_genes_detected'] = np.sum(boolean_expression, axis=1)
    # fraction mitochondrial transcripts per cell
    mito_genes = [gene for gene in adata.var.index 
                  if gene.lower().startswith('mt-')]
    # conversion to array in line below is necessary if adata.X is sparse
    adata.obs['mito_frac'] = np.array(np.sum(
        adata[:,mito_genes].X, axis=1)).flatten() / adata.obs['total_counts']
    # fraction ribosomal transcripts per cell
    ribo_genes = [gene for gene in adata.var.index 
                  if (gene.lower().startswith('rpl') 
                      or gene.lower().startswith('rps'))]
    adata.obs['ribo_frac'] = np.array(np.sum(
        adata[:,ribo_genes].X, axis=1)).flatten() / adata.obs['total_counts']
    # cell complexity (i.e. number of genes detected / total transcripts)
    adata.obs['compl'] = adata.obs['n_genes_detected']\
    / adata.obs['total_counts']
    # genes
    adata.var['n_cells'] = np.sum(boolean_expression, axis=0).T
    return adata

def get_sample_annotation_table_LCA(
    project_dir="/storage/groups/bcf/datasets/SingleCell/10x/fastq/theis/lung_cell_atlas/",
):
    """reads in metadata as collected throught LCA_metadata tables. 
    args:
        project_dir - path to directory that has one directory per dataset, 
        and as subdirectory 01_Metadata, in which LCA_metadata_....csv are 
        stored. 
    Returns:
        pandas dataframe with one row per sample, and all matching metadata."""
    # get paths to all LCA metatables available
    file_paths_long = glob.glob("{}*/01_Metadata/LCA_metadata*.csv".format(project_dir))
    # store paths together with file names (some files are stored in two different
    # places, since the same lab had multiple datasets, and they filled out only
    # one table)
    path_to_name_dir = {
        file_path: file_path.split("/")[-1] for file_path in file_paths_long
    }
    files_read = list()
    meta_tables = dict()
    # store metatables for each unique file name
    for file_path, file_name in path_to_name_dir.items():
        if file_name not in files_read:
            print(file_name)
            # read csv
            meta = pd.read_csv(file_path, index_col=2)
            # remove example rows
            meta = meta.loc[[inst != "EXAMPLE INSTITUTE" for inst in meta.Institute], :]
            # store
            meta_tables[file_path] = meta
            files_read.append(file_name)
    # merge tables into one
    metadata = pd.concat(meta_tables.values())
    # remove rows with NaN as index
    print(
        "number of rows without rowname/sample name (will be removed):",
        sum(metadata.index.isnull()),
    )
    metadata = metadata.loc[metadata.index.isnull() == False, :]
    # check if sample ids abd donor ids are unique
    print(
        "Sample IDs unique?", len(metadata.index) == len(set(metadata.index.tolist()))
    )
    # check which rows have no donor_ID
    #     metadata.loc[metadata.subject_ID.isnull(), :]
    # based on the notes they added, we will add donor IDs:
    sample_to_donor_lafyatis = {
        "SC14": "pitt_donor_1",
        "SC31": "pitt_donor_2",
        "SC31D": "pitt_donor_2",
        "SC45": "pitt_donor_3",
        "SC56": "pitt_donor_4",
        "SC_56": "pitt_donor_5",
        "SC59": "pitt_donor_6",
        "SC155": "pitt_donor_7",
        "SC156": "pitt_donor_7",
    }
    # now map samples to donors
    sample_to_donor = {
        sample: donor for sample, donor in zip(metadata.index, metadata.subject_ID)
    }
    for sample, donor in sample_to_donor_lafyatis.items():
        sample_to_donor[sample] = donor
    metadata.subject_ID = metadata.index.map(sample_to_donor)
    # print number of samples without donor ID
    print("Number of samples without donor ID:", sum(metadata.subject_ID.isnull()))
    # return result
    return metadata


def add_sample_annotations(
    adata,
    path_to_metadata_file="/home/icb/lisa.sikkema/Documents/LCA/metadata_harmonized/metadata_for_Lung_Cell_Atlas_20200507.csv",
):
    """adds sample/donor annotation to AnnData.obs, such as age, sex and smoking info.
    The function uses the metadata.csv file that we designed for LCA.
    It moreover does not overwrite values that already exist in AnnData.obs,
    but only replaces "nan" or NaN values. Outputs an annotated AnnData object."""
    # read in metadata, with one sample per row ideally. However, e.g. Martijn's
    # entries have one patient per row. Therefore we will exclude entries here
    # for sample-specific columns such as anatomical region, see futher below.
    meta_sample = pd.read_csv(path_to_metadata_file, index_col=0,)
    # rename columns we plan to use, so that they correspond with names from our
    # AnnData.obs columns:
    meta_sample.rename(
        columns={
            "age, in years": "age",
            "age, range": "age_range",
            "mixed ethnicity": "ethnicity_mixed",
            "smoking status": "smoking",
            "smoking history": "pack_years",
            "anatomical region coarse": "anatomical_region",
            "anatomical region detailed": "anatomical_region_detailed",
            "subject type": "subject_type",
            "subject_ID_as_published": "donor" # LISA TEST LISA
        },
        inplace=True,
    )
    # summarize this table, so that we only have the categories we're interested in
    sample_info = meta_sample.groupby("Sample_ID").agg(
        {
            "age": "first",
            "age_range": "first",
            "sex": "first",
            "ethnicity": "first",
            "ethnicity_mixed": "first",
            "smoking": "first",
            "pack_years": "first",
            "anatomical_region": "first",
            "anatomical_region_detailed": "first",
            "subject_type": "first",
            "donor": "first" # LISA TEST LISA
        }
    )
    # Now match rows from sample_info/metadata table with our anndata annotation.
    # This is a bit complicated, because in some cases 'SampleID' matches with
    # our AnnData sample names, and in other cases it matches with our AnnData
    # donor names. We will deal with that uncertainty below.
    # first, relace np.nan with None, so that sorting of donor names is possible:
    adata.obs.donor = adata.obs.donor.where(pd.notnull(adata.obs.donor), None)
    # check if all "donors" are either as sample name or donor name
    # in adata. Print those donors that don't have a match in our
    # AnnData object:
    donors_in_adata = sorted(set(d for d in adata.obs.donor if d != None))
    samples_in_adata = sorted(set(adata.obs["sample"]))
    idc_not_traced = list()
    idc_with_sample_transl = list()
    idc_with_donor_transl = list()
    for idx in sample_info.index:
        if idx in donors_in_adata:
            idc_with_donor_transl.append(idx)
        elif idx in samples_in_adata:
            idc_with_sample_transl.append(idx)
        else:
            idc_not_traced.append(idx)
    if len(idc_not_traced) == 0:
        print("All donors were found in either adata.obs.donor or adata.obs.sample")
    else:
        print(
            "These donors from the imported metadata table were not found in your adata.obs.donor nor adata.obs['sample']:"
        )
        print(idc_not_traced)
    # now that we know which SampleIDs corresponded to donors, we will make
    # dictionaries, one per annotation category, that translate donor to the
    # correct annotation label:
    # make donor to annotation dict:
    donor_to_annotation_dict = dict()
    for ann_cat in sample_info.columns:
        donor_to_annotation_dict[ann_cat] = dict(
            zip(idc_with_donor_transl, sample_info.loc[idc_with_donor_transl, ann_cat])
        )
        # note that anatomical region for ARMS samples is incorrect in the table
        # so take that out
        if ann_cat in ["anatomical region coarse", "anatomical region detailed"]:
            # store keys that need to be deleted:
            keys_to_delete = [
                key for key in donor_to_annotation_dict[ann_cat].keys() if "ARMS" in key
            ]
            for key in keys_to_delete:
                del donor_to_annotation_dict[ann_cat][key]
    # now we'll do the same for SampleIDs in the metadata/sampleinfo df that have
    # a corresponding sample (rather than donor) in our AnnData object:
    sample_to_annotation_dict = dict()
    for ann_cat in sample_info.columns:
        sample_to_annotation_dict[ann_cat] = dict(
            zip(
                idc_with_sample_transl, sample_info.loc[idc_with_sample_transl, ann_cat]
            )
        )
    # We now design a function that takes in three annotation vectors, and outputs
    # for each index the first entry that is not "nan" or NaN. It is ordered,
    # so if the first entry is not NaN, it will ignore the remaining entries.
    # We will take the AnnData entries as the first and therefore "overwriting"
    # one, since it already contains annotations for samples that are not in our
    # metadata file. We want to retain those annotations!
    def output_first_nonNaN_value(value_1, value_2, value_3):
        """takes in three values, and returns the first non-nan value"""
        for value in [value_1, value_2, value_3]:
            if isinstance(value, str):
                if value == "nan":
                    pass
                else:
                    return value
            if isinstance(value, float):
                if np.isnan(value):
                    pass
                else:
                    return value
        # if nothing has resulted in returning of value, return the final value...
        return value_3

    # We will now take the original ann_cat vector if existent, i.e. the one that
    # we already have in AnnData. Then we add the vector that has translated
    # samples-to-annotation labels, and the vector that has translated
    # donor-to-annotation labels. We'll merge those using the function above,
    # and assign it to AnnData.obs.
    for ann_cat in sample_info.columns:
        print(ann_cat)
        if ann_cat in adata.obs:
            original_ann = adata.obs[ann_cat]
        else:
            original_ann = pd.Series(adata.shape[0] * [np.nan])
        sample_ann = adata.obs["sample"].map(sample_to_annotation_dict[ann_cat])
        donor_ann = adata.obs.donor.map(donor_to_annotation_dict[ann_cat])
        adata.obs[ann_cat] = [
            output_first_nonNaN_value(val1, val2, val3)
            for val1, val2, val3 in zip(
                original_ann.values, sample_ann.values, donor_ann.values
            )
        ]
    # return resulting anndata:
    return adata

def plot_QC(anndata_dict, project_name, scale=1):
    n_cols = 4
    n_rows = len(anndata_dict.keys())
    fig = plt.figure(figsize=(n_cols * 3 * scale, n_rows * 2.5 * scale))
    plt.suptitle('QC ' + project_name, fontsize=16, y=1.02)
    title_fontsize = 8
    ax_dict = dict()
    ax = 1
    for sample_ID in sorted(anndata_dict.keys()):
        adata = anndata_dict[sample_ID]
        # total counts
        ax_dict[ax] = fig.add_subplot(n_rows, n_cols, ax)
        ax_dict[ax].hist(adata.obs['log10_total_counts'], bins=50, range=(2,5))
        ax_dict[ax].set_title('log10 total counts per cell', 
                              fontsize=title_fontsize)
        ax_dict[ax].set_xlabel('log10 total counts')
        ax_dict[ax].set_ylabel(sample_ID, fontsize=16)
        ax = ax + 1
        # mitochondrial transcript percentage
        ax_dict[ax] = fig.add_subplot(n_rows, n_cols, ax)
        ax_dict[ax].hist(adata.obs['mito_frac'], bins=50, range=(0,1))
        ax_dict[ax].set_title('fraction of mitochondrial transcripts/cell', 
                      fontsize=title_fontsize)
        ax_dict[ax].set_xlabel('fraction of mito transcripts')
        ax_dict[ax].set_ylabel('frequency')
        ax = ax + 1
        # log10 number of genes detected
        ax_dict[ax] = fig.add_subplot(n_rows, n_cols, ax)
        ax_dict[ax].hist(np.log10(adata.obs['n_genes_detected']), 
                         bins=50, range=(2,5))
        ax_dict[ax].set_title('log10 number of genes detected', 
                              fontsize=title_fontsize)
        ax_dict[ax].set_xlabel('log10 number of genes')
        ax_dict[ax].set_ylabel('frequency')
        ax = ax + 1
        # cell complexity, scatter
        ax_dict[ax] = fig.add_subplot(n_rows, n_cols, ax)
        mappable_ax = str(ax) + '_m'
        ax_dict[mappable_ax] = ax_dict[ax].scatter(adata.obs['log10_total_counts'],
                    np.log10(adata.obs['n_genes_detected']), 
                    c=adata.obs['mito_frac'], s=1)
        ax_dict[ax].set_title('cell complexity \ncolored by mitochondrial fraction', 
                      fontsize=title_fontsize)
        ax_dict[ax].set_xlabel('log10 total counts')
        ax_dict[ax].set_ylabel('log10 number of genes detected')
        fig.colorbar(mappable=ax_dict[mappable_ax],ax=ax_dict[ax])
        ax = ax + 1
    plt.tight_layout()
    plt.show()
    return fig


def age_converter(age, age_range, verbose=False):
    """takes in two values (age and age range (format: [number]-[number])) 
    where only one has a value, and the other is NaN. Returns 
    either the original age or the mean of the age range."""
    if isinstance(age, float):
        if np.isnan(age):
            if verbose:
                print(age_range)
            age_lower = np.float(age_range.split("-")[0])
            age_higher = np.float(age_range.split("-")[1])
            mean_age = np.mean([age_lower, age_higher])
            if verbose:
                print(mean_age)
            return mean_age
        else:
            return age
    else:
        return age


def metadata_cleaner(meta_df):
    """Takes in dataframe of LCA metadata (e.g. adata.obs). Remaps categories that 
    are listed in remapper_dicts. Returns cleaned df."""
    # create remapping dictionaries
    remapper_dicts = dict()
    remapper_dicts["known lung disease"] = {"NO": "no", "no": "no", "yes": "yes"}
    remapper_dicts["smoking status"] = {
        "former": "former",
        "active": "active",
        "never": "never",
        "past": "former",
    }
    remapper_dicts["cell ranger version "] = {
        "1.3.1": "1.3.1",
        "2": "2",
        "2.0.1": "2.0.1",
        "2.0.2": "2.0.2",
        "3": "3",
        "3.1.0": "3.1.0",
        "3.0.2": "3.0.2",
        "CellRanger software version 2.0.2": "2.0.2",
    }
    remapper_dicts["condition"] = {
        'asthma': 'asthma',
        'carcinoid': 'carcinoid',
        'childhood asthma': 'childhood asthma',
        'healthy': 'healthy',
        'healthy_volunteer': 'healthy',
        'non-small cell lung cancer': 'non-small cell lung cancer'
    }
    remapper_dicts["reference genome coarse"] = {
        'GRCh38': 'GRCh38',
        'GRCh38 1.2.0': 'GRCh38',
        'GRCh38 1.2.1': 'GRCh38',
    }
    remapper_dicts["subject type"] = {
        'alive_healthy': 'alive_healthy',
        'alive_disease': 'alive_disease',
        'healthy_volunteer': 'alive_healthy',
        'organ_donor': 'organ_donor',
        'volunteer': 'alive_healthy'
    }
    remapper_dicts["sample type"] = {
        "Biopsy": "biopsy",
        "Brushing": "brushing",
        "brushing":"brushing",
        "biopsy": "biopsy",
        "brush": "brushing",
        "donor_lung": "donor_lung",
        "surgical_resection": "surgical_resection",
    }
    remapper_dicts["ethnicity"] = {
        "Indian Sub-continent": "asian",
        "asian": "asian",
        "black": "black",
        "latino": "latino",
        "mixed": "mixed",
        "nan": "nan",
        "pacific islander": "pacific islander",
        "white": "white",
    }
    # known lung disease
    for cat, mapping in remapper_dicts.items():
        if cat in meta_df.columns:
            original_cats = set(meta_df[cat])
            # check if all categories in data are represented
            original_cats = [x for x in original_cats if not check_if_nan(x)]
            if not set(original_cats).issubset(set(mapping.keys())):
                print("original cats:", original_cats)
                print("remapping cats:", set(mapping.keys()))
                print(
                    "remap dict is underdefined! (fewer cats than in meta_df) Exiting."
                )
                return
            meta_df[cat] = meta_df[cat].map(mapping)
    return meta_df

# HELPER FUNCTIONS


# helper function for get_gene_renamer_dict function
def rreplace(s, old, new, occurrence, only_if_not_followed_by=None):
    """replaces occurences of character, counted from last to first, with new 
    character. 
    Arguments:
    s - string
    old - character/string to be replaced
    new - character/string to replace old with
    occurence - nth occurence up to which instances should be replaced, counting 
    from end to beginning. e.g. if set to 2, the two last instances of "old" 
    will be replaced
    only_if_not_followed_by - if this is set to a string, [old] is only replaced
    by [new] if [old] is not followed by [only_if_not_followed_by] in [s]. Useful
    for gene names like abc-1.d, in that case we do not want to replace the dash.
    Returns:
    string with replaced character
    """
    if old not in s:
        return s
    elif only_if_not_followed_by != None:
        # test if latest instance of old is followed by [only_if_not_followed_by]
        last_inst_old = s.rfind(old)
        last_inst_oinfb = s.rfind(only_if_not_followed_by)
        if last_inst_oinfb > last_inst_old:
            return s
    li = s.rsplit(old, occurrence)
    return new.join(li)
