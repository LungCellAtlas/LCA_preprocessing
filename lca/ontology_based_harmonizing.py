# THIS SCRIPT SERVES TO TRANSLATE INDIVIDUAL DATASET ANNOTATIONS TO CELL TYPE ANNOTATIONS AS IMPLEMENTED BY SASHA AND MARTIJN! :)
import pandas as pd
import numpy as np
import scanpy as sc
import utils # this is a custom script from me

def nan_checker(entry):
	"""replaces entry that is not a cell type label, but rather some other
	entry that exists in the table (e.g. |, >, nan) with None. Retains
	labels that look like proper labels."""
	if entry == '|':
		new_entry = None
	elif entry == '>':
		new_entry = None
	elif pd.isna(entry):
		new_entry = None
	else:
		new_entry = entry
	return new_entry

def load_harmonizing_table(path_to_csv):
	"""Loads the csv download version of the google doc in which 
	cell types for each dataset are assigned a consensus equivalent.
	Returns the cleaned dataframe"""
	cell_type_harm_df = pd.read_csv(
		path_to_csv, 
		header=1
		)
	cell_type_harm_df = cell_type_harm_df.applymap(nan_checker)
	return cell_type_harm_df

def consensus_index_renamer(consensus_df, idx):
    highest_res = int(float(consensus_df.loc[idx, "highest_res"]))
    new_index = (
        str(highest_res) + "_" + consensus_df.loc[idx, "level_" + str(highest_res)]
    )
    return new_index

def create_consensus_table(harmonizing_df, max_level=5):
    """creates a clean consensus table based on the harmonizing_df that also 
    includes dataset level info. The output dataframe contains, for each consensus
    cell type, it's 'parent cell types', an indication of whether the cell 
    type is considered new, and the level of the annotation.
    max_level - maximum level of annotation available"""
    # set up empty consensus df, that will be filled later on
    consensus_df = pd.DataFrame(
        columns=[
            'level_' + str(levnum) for levnum in range(1,max_level + 1)
        ] + [
            'highest_res','new','harmonizing_df_index'
        ]
    )
    # store the index of the harmonizing df, so that we can match rows later on
    consensus_df["harmonizing_df_index"] = harmonizing_df.index
    # create dictionary storing the most recent instance of every level,
    # so that we always track the 'mother levels' of each instance
    most_recent_level_instance = dict()
    # loop through rows
    for idx in harmonizing_df.index:
        # set 'new' to false. This variable indicates if the cell type annotation
        # is new according to Sasha and Martijn. It will be checked/changed later.
        new = False
        # create a dictionary where we story the entries of this row,
        # for each level
        original_entries_per_level = dict()
        # create a dictionary where we store the final entries of this row,
        # for each level. That is, including the mother levels.
        final_entries_per_level = dict()
        # if all entries in this row are None, continue to next row:
        if (
            sum(
                [
                    pd.isna(x)
                    for x in harmonizing_df.loc[
                        idx, ["Level_" + str(levnum) for levnum in range(1, max_level + 1)]
                    ]
                ]
            )
            == max_level
        ):
            continue
        # for each level, check if we need to store the current entry,
        # the mother level entry, or no entry (we do not want to store daughters)
        for level in range(1, max_level + 1):
            original_entry = harmonizing_df.loc[idx, "Level_" + str(level)]
            # if the current level has an entry in the original dataframe,
            # update the 'most_recent_level_instance'
            # and store the entry as the final output entry for this level
            if original_entry != None:
                # store the lowest level annotated for this row:
                lowest_level = level

                # check if entry says 'New!', then we need to obtain actual label
                # from the dataset where the label appears
                if original_entry == "New!":
                    # set new to true now
                    new = True
                    # get all the unique entries from this row
                    # this should be only one entry aside from 'None' and 'New!'
                    actual_label = list(set(harmonizing_df.loc[idx, :]))
                    # filter out 'New!' and 'None'
                    actual_label = [
                        x for x in actual_label if not pd.isna(x) and x != "New!"
                    ][0]
                    original_entry = actual_label
                # update most recent instance of level:
                most_recent_level_instance[level] = original_entry
                # store entry as final entry for this level
                final_entries_per_level[level] = original_entry
                # Moreover, store the most recent instances of the lower (parent)
                # levels as final output:
                for parent_level in range(1, level):
                    final_entries_per_level[parent_level] = most_recent_level_instance[
                        parent_level
                    ]
                # and set the daughter levels to None:
                for daughter_level in range(level + 1, max_level + 1):
                    final_entries_per_level[daughter_level] = None
                break
        # if none of the columns have an entry,
        # set all of them to None:
        if bool(final_entries_per_level) == False:
            for level in range(1, 5):
                final_entries_per_level[level] = None
        # store the final outputs in the dataframe:
        consensus_df.loc[idx, "highest_res"] = int(lowest_level)
        consensus_df.loc[idx, "new"] = new
        consensus_df.loc[idx, consensus_df.columns[:max_level]] = [
            final_entries_per_level[level] for level in range(1, max_level + 1)
        ]
    rows_to_keep = (
        idx
        for idx in consensus_df.index
        if not consensus_df.loc[
            idx, ["level_" + str(levnum) for levnum in range(1, max_level + 1)]
        ]
        .isna()
        .all()
    )
    consensus_df = consensus_df.loc[rows_to_keep, :]
    # rename indices so that they also specify the level of origin
    # this allows for indexing by cell type while retaining uniqueness
    # (i.e. we're able to distinguis 'Epithelial' level 1, and level 2)
    consensus_df.index = [
        consensus_index_renamer(consensus_df, idx) for idx in consensus_df.index
    ]
    # Now check if there are double index names. Sasha added multiple rows
    # for the same consensus type to accomodate his cluster names. We
    # should therefore merge these rows:
    # First create an empty dictionary in which to store the harmonizing_df row
    # indices of the multiple identical rows:
    harmonizing_df_index_lists = dict()
    for unique_celltype in set(consensus_df.index):
        # if there are multiple rows, the only way in which they should differ
        # is their harmonizing_df_index. Check this:
        celltype_sub_df = consensus_df.loc[unique_celltype]
        if type(celltype_sub_df) == pd.DataFrame and celltype_sub_df.shape[0] > 1:
            # check if all levels align:
            for level in range(1, max_level + 1):
                if len(set(celltype_sub_df["level_" + str(level)])) > 1:
                    print(
                        "WARNING: {} has different annotations in different rows at level {}. Look into this!!".format(
                            unique_celltype, str(level)
                        )
                    )
            harmonizing_df_index_list = celltype_sub_df["harmonizing_df_index"].values
        # if there was only one index equal to the unique celltype, the
        # celltype_sub_df is actually a pandas series and we need to index
        # it diffently
        else:
            harmonizing_df_index_list = [celltype_sub_df["harmonizing_df_index"]]
        # store the harmonizing_df_index_list in the consensus df
        # we use the 'at' function to be able to store a list in a single
        # cell/entry of the dataframe.
        harmonizing_df_index_lists[unique_celltype] = harmonizing_df_index_list
    # now that we have a list per cell type, we can drop the duplicate columns
    consensus_df.drop_duplicates(
        subset=["level_" + str(levnum) for levnum in range(1, max_level + 1)], inplace=True
    )
    # and replace the harmonizing_df_index column with the generated complete lists
    consensus_df["harmonizing_df_index"] = None
    for celltype in consensus_df.index:
        consensus_df.at[celltype, "harmonizing_df_index"] = harmonizing_df_index_lists[
            celltype
        ]
    #     forward propagate coarser annotations
    #     add a prefix (matching with the original level of coarseness) to the
    #     forward-propagated annotations
    for celltype in consensus_df.index:
        highest_res = consensus_df.loc[celltype, "highest_res"]
        # go to next level of loop if the highest_res is nan
        if not type(highest_res) == pd.Series:
            if pd.isna(highest_res):
                continue
        for i in range(highest_res + 1, max_level + 1):
            consensus_df.loc[celltype, "level_" + str(i)] = celltype
    return consensus_df

# def create_dataset_translation_tables(
#     adata, consensus_df, harmonizing_df, verbose=True, limited_dataset_list=None
# ):
#     """returns a dictionary with a dataframe for each dataset. The indices 
#     of the dataframe correspond to original labels from the dataset,
#     the columns contain information on the consensus translation. I.e. a 
#     translation at each level, whether or not the cell type is 'new', and what
#     the level of the annotation is."""
#     # set number of annotation levels:
#     number_of_levels = 5
#     # list the datasets of interest. This name should correspond to the
#     # dataset name in the anndata object, 'dataset' column.
#     if limited_dataset_list == None:
#         datasets = [
#             "Habermann",
#             "Madissoon",
#             "VieiraBraga",
#             "Morse",
#             "Reyfman",
#             "regev_healthy_nucseq",
#             "whitsett_xu",
#             "Misharin_new",
#         ]
#     else:
#         datasets = limited_dataset_list
#     # store the column name in adata.obs that contains the cell type annotations
#     # for each dataset, in a dictionary:
#     adata_celltype_colnames = dict()
#     adata_celltype_colnames["Habermann"] = "Habermann_celltype"
#     adata_celltype_colnames["Madissoon"] = "Madissoon_Celltypes"
#     adata_celltype_colnames["VieiraBraga"] = "VieiraBraga_ClusterNumbers"
#     adata_celltype_colnames["VieiraBraga_asthma"] = "vieirabraga_celltype"
#     adata_celltype_colnames["Morse"] = "Morse_annotation"
#     adata_celltype_colnames["Reyfman"] = "Reyfman_annotations"
#     adata_celltype_colnames["regev_healthy_nucseq"] = "regev_celltype"
#     adata_celltype_colnames["tsankov"] = "tsankov_celltype"
#     adata_celltype_colnames["whitsett_xu"] = "whitsett_xu_celltype"
#     adata_celltype_colnames["Misharin_new"] = "Misharin_celltype"
#     adata_celltype_colnames["kaminski"] = "kaminski_celltype"
#     adata_celltype_colnames["krasnow"] = "krasnow_celltype"
#     adata_celltype_colnames["schultze"] = "schultze_celltype"
#     adata_celltype_colnames["schultze_falk"] = "schultze_falk_celltype"
#     adata_celltype_colnames["rawlins"] = "rawlins_celltype"
#     adata_celltype_colnames["xavier"] = "xavier_celltype"
#     adata_celltype_colnames["meyer_new"] = "meyer_new_celltype"
#     adata_celltype_colnames["spence_fetal"] = "spence_celltype"
#     adata_celltype_colnames["barbry"] = "barbry_celltype"
#     adata_celltype_colnames["seibold"] = "seibold_celltype"
#     adata_celltype_colnames["shalek"] = "shalek_celltype"
#     adata_celltype_colnames["VieiraBraga_nasal"] = "nawijn_celltype"
#     adata_celltype_colnames["linnarsson"] = "linnarsson_celltype"

#     # adata_celltype_colnames["krasnow_ss2"] = "krasnow_celltype"
#     # store the column name in the "cell type harmonization table" (google doc)
#     # that corresponds to the dataset, in a dictionary:
#     cell_type_harm_colnames = dict()
#     cell_type_harm_colnames["Habermann"] = "Habermann_2019"
#     cell_type_harm_colnames["Madissoon"] = "Madissoon_2020"
#     cell_type_harm_colnames["VieiraBraga"] = "Vieira-Braga_2019_F3_4"
#     cell_type_harm_colnames["VieiraBraga_asthma"] = "Vieira-Braga_2019_F3_4"
#     cell_type_harm_colnames["Morse"] = "Morse_2019"
#     cell_type_harm_colnames["Reyfman"] = "Reyfman_2019"
#     cell_type_harm_colnames["regev_healthy_nucseq"] = "Regev"
#     cell_type_harm_colnames["tsankov"] = "Tsankov"
#     cell_type_harm_colnames["whitsett_xu"] = "Whitsett_Xu"
#     cell_type_harm_colnames["Misharin_new"] = "NU_CZI01"
#     cell_type_harm_colnames["kaminski"] = "Kaminski"
#     cell_type_harm_colnames["krasnow"] = "Krasnow"
#     cell_type_harm_colnames["schultze"] = "Schultze"
#     cell_type_harm_colnames["schultze_falk"] = "Schultze_Falk"
#     cell_type_harm_colnames["rawlins"] = "Rawlins_fetal"
#     cell_type_harm_colnames["xavier"] = "Xavier"
#     cell_type_harm_colnames["meyer_new"] = "Meyer_new"
#     cell_type_harm_colnames["spence_fetal"] = "Spence_fetal"
#     cell_type_harm_colnames["barbry"] = "Deprez_2020"
#     cell_type_harm_colnames["seibold"] = "Seibold"
#     cell_type_harm_colnames["shalek"] = "Shalek"
#     cell_type_harm_colnames["VieiraBraga_nasal"] = "VieiraBraga_UMCG_nasal"
#     cell_type_harm_colnames["linnarsson"] = "Linnarsson_temp"
#     # cell_type_harm_colnames["krasnow_ss2"] = "Krasnow"
#     translation_dfs = dict()
#     for dataset in datasets:
#         if verbose:
#             print("working on dataset " + dataset + "...")
#         cell_type_harm_column = cell_type_harm_colnames[dataset]
#         # adata_column = "ClusterNumbers_VieiraBraga"
#         adata_celltype_column = adata_celltype_colnames[dataset]

#         # get cell type names, remove nan and None from list:
#         cell_types_original = sorted(
#             [
#                 x
#                 for x in set(adata.obs[adata_celltype_column])
#                 if str(x) != "nan" and str(x) != "None"
#             ]
#         )
#         translation_df = pd.DataFrame(
#             columns=consensus_df.columns, index=cell_types_original
#         )
#         for label in cell_types_original:
#             print(label)
#             # if the label is 'nan', skip this row
#             if label == "nan":
#                 continue
#             # get the rows in which this label appears in the dataset column
#             # of the harmonizing_df:
#             ref_rows = harmonizing_df[cell_type_harm_column][
#                 harmonizing_df[cell_type_harm_column] == label
#             ].index.tolist()
#             # if the label does not occur, there is a discrepancy between the
#             # labels in adata and the labels in the harmonizing table made
#             # by Martijn and Sasha.
#             if len(ref_rows) == 0 and verbose:
#                 print(
#                     "HEY THERE ARE NO ROWS IN THE harmonizing_df WITH THIS LABEL ({}) AS ENTRY!".format(
#                         label
#                     )
#                 )
#             # if the label occurs twice, there is some ambiguity as to how to
#             # translate this label to a consensus label. In that case, translate
#             # to the finest level that is identical in consensus translation
#             # between the multiple rows.
#             elif len(ref_rows) > 1 and verbose:
#                 print(
#                     "HEY THE NUMBER OF ROWS WITH ENTRY {} IS NOT 1 but {}!".format(
#                         label, str(len(ref_rows))
#                     )
#                 )
#                 # get matching indices from consensus_df:
#                 consensus_idc = list()
#                 for i in range(len(ref_rows)):
#                     consensus_idc.append(
#                         consensus_df[
#                             [
#                                 ref_rows[i] in index_list
#                                 for index_list in consensus_df["harmonizing_df_index"]
#                             ]
#                         ].index[0]
#                     )
#                 # now get the translations for both cases:
#                 df_label_subset = consensus_df.loc[consensus_idc, :]
#                 # store only labels that are common among all rows with this label:
#                 for level in range(1, number_of_levels + 1):
#                     col = "level_" + str(level)
#                     n_translations = len(set(df_label_subset.loc[:, col]))
#                     if n_translations == 1:
#                         # update finest annotation
#                         finest_annotation = df_label_subset.loc[consensus_idc[0], col]
#                         # update finest annotation level
#                         finest_level = level
#                         # store current annotation at current level
#                         translation_df.loc[label, col] = finest_annotation

#                     # if level labels differ between instances, store None for this level
#                     else:
#                         translation_df.loc[label, col] = (
#                             str(finest_level) + "_" + finest_annotation
#                         )
#                         translation_df.loc[label, "highest_res"] = finest_level
#                 # set "new" to false, since we fell back to a common annotation:
#                 translation_df.loc[label, "new"] = False
#                 # add ref rows to harmonizing_df_index?
#                 # ...
#             # if there is only one row with this label, copy the consensus labels
#             # from the same row to the translation df.
#             else:
#                 consensus_idx = consensus_df[
#                     [
#                         ref_rows[0] in index_list
#                         for index_list in consensus_df["harmonizing_df_index"]
#                     ]
#                 ]
#                 consensus_idx = consensus_idx.index[
#                     0
#                 ]  # loc[label,'harmonizing_df_index'][0]
#                 translation_df.loc[label, :] = consensus_df.loc[consensus_idx, :]
#             # store the translation df of this dataset in the dictionary
#             translation_dfs[dataset] = translation_df
#     if verbose:
#         print("Done!")
#     return translation_dfs

# def consensus_annotate_anndata(
#     adata, translation_dfs, verbose=True, limited_dataset_list=None
# ):
#     """annotates cells in adata with consensus annotation. Returns adata."""
#     # list the datasets of interest. This name should correspond to the 
#     # dataset name in the anndata object, 'dataset' column.
#     if limited_dataset_list == None:
#         datasets = [
#             "Habermann",
#             "Madissoon",
#             "VieiraBraga",
#             "Morse",
#             "Reyfman",
#             "regev_healthy_nucseq",
#             "whitsett_xu",
#             "Misharin_new"
#         ]
#     else:
#         datasets = limited_dataset_list
#     # specify the columns to copy:
#     col_to_copy = [
#         "level_1",
#         "level_2",
#         "level_3",
#         "level_4",
#         "level_5",
#         "highest_res",
#         "new",
#     ]
#     # add 'ann_' prefix for in adata.obs:
#     col_to_copy_new_names = ["ann_" + name for name in col_to_copy]
#     # add these columns to adata:
#     for col in col_to_copy_new_names:
#         adata.obs[col] = None
#     # store the column name in adata.obs that contains the cell type annotations
#     # for each dataset, in a dictionary:
#     adata_celltype_colnames = dict()
#     adata_celltype_colnames["Habermann"] = "Habermann_celltype"
#     adata_celltype_colnames["Madissoon"] = "Madissoon_Celltypes"
#     adata_celltype_colnames["VieiraBraga"] = "VieiraBraga_ClusterNumbers"
#     adata_celltype_colnames["VieiraBraga_asthma"] = "vieirabraga_celltype"
#     adata_celltype_colnames["Morse"] = "Morse_annotation"
#     adata_celltype_colnames["Reyfman"] = "Reyfman_annotations"
#     adata_celltype_colnames["regev_healthy_nucseq"] = "regev_celltype"
#     adata_celltype_colnames["tsankov"] = "tsankov_celltype"
#     adata_celltype_colnames["whitsett_xu"] = "whitsett_xu_celltype"
#     adata_celltype_colnames["Misharin_new"] = "Misharin_celltype"
#     adata_celltype_colnames["kaminski"] = "kaminski_celltype"
#     adata_celltype_colnames["krasnow"] = "krasnow_celltype"
#     adata_celltype_colnames["schultze"] = "schultze_celltype"
#     adata_celltype_colnames["schultze_falk"] = "schultze_falk_celltype"
#     adata_celltype_colnames["rawlins"] = "rawlins_celltype"
#     adata_celltype_colnames["xavier"] = "xavier_celltype"
#     adata_celltype_colnames["meyer_new"] = "meyer_new_celltype"
#     adata_celltype_colnames["spence_fetal"] = "spence_celltype"
#     adata_celltype_colnames["barbry"] = "barbry_celltype"
#     adata_celltype_colnames["seibold"] = "seibold_celltype"
#     adata_celltype_colnames["shalek"] = "shalek_celltype"
#     adata_celltype_colnames["VieiraBraga_nasal"] = "nawijn_celltype"
#     adata_celltype_colnames["linnarsson"] = "linnarsson_celltype"

#     for dataset in datasets:
#         if verbose:
#             print(dataset)
#         # get matching column name in adata
#         adata_celltype_col = adata_celltype_colnames[dataset]
#         # get translation df matching with dataset
#         translation_df = translation_dfs[dataset]
#         # loop through cell types from dataset (as they appear in the anndata
#         # object)
#         dataset_celltypes = sorted(
#             [
#                 x
#                 for x in set(adata.obs[adata_celltype_col])
#                 if str(x) != "nan" and str(x) != "None"
#             ]
#         )
#         for cell_type in dataset_celltypes:
#             if verbose:
#                 print(cell_type)
#             # get indices of cells from the dataset under consideration,
#             # annotated with cell type under consideration
#             # since whitsett_xu is actually two datasets (10x versus 
#             # dropseq), the conditional is slightly different there.
#             # the same holds for krasnow (10x and ss2)
#             if dataset == 'whitsett_xu':
#                 idc = [
#                     x
#                     for x in adata.obs.index
#                     if adata.obs.loc[x, "dataset"] in [
#                     'Xu_Whitsett_Peds_10X', 'Xu_Whitsett_Peds_dropseq']
#                     and adata.obs.loc[x, adata_celltype_col] == cell_type
#                 ] 
#             elif dataset == 'krasnow':
#                 idc = [
#                     x
#                     for x in adata.obs.index
#                     if adata.obs.loc[x, "dataset"] in [
#                     'krasnow_10x', 'krasnow_ss2']
#                     and adata.obs.loc[x, adata_celltype_col] == cell_type
#                 ] 
#             # for all other datasets, simply check if adata.obs 
#             # dataset corresponds to dataset name:
#             else:
#                 idc = [
#                     x
#                     for x in adata.obs.index
#                     if adata.obs.loc[x, "dataset"] == dataset
#                     and adata.obs.loc[x, adata_celltype_col] == cell_type
#                 ]
#             # set the consensus columns of these cells appropriately:
#             for i in range(len(col_to_copy)):
#                 col_name = col_to_copy[i]
#                 new_col_name = col_to_copy_new_names[i]
#                 adata.obs.loc[idc, new_col_name] = translation_df.loc[
#                     cell_type, col_name
#                 ]
#             #             adata.obs.loc[idc, "ann_new"] = translation_df.loc[cell_type, "new"]
#     #             for level in range(1, 5):
#     #                 adata.obs.loc[idc, "ann_level_" + str(level)] = translation_df.loc[
#     #                     cell_type, "level_" + str(level)
#     #                 ]
#     return adata
  
# NEW FUNCTIONS, THAT ARE CLEANER THAN THE ONES ABOVE AND MATCH WITH THE WAY I'M ANNOTATING DATA FOR LCA (AS OPPOSED TO COVID19 PROJECT)
def create_celltype_to_consensus_translation_df(
    adata,
    consensus_df,
    harmonizing_df,
    verbose=True,
    number_of_levels=5,
    ontology_type="cell_type",
):
    """returns a dataframe. The indices of the dataframe correspond to original
    labels from the dataset, with the dataset name as a prefix (as it appears in
    the input adata.obs.dataset), the columns contain information on the
    consensus translation. I.e. a translation at each level, whether or not the
    cell type is 'new', and what the level of the annotation is.
    ontology_type <"cell_type","anatomical_region_coarse","anatomical_region_fine"> - type of ontology
    """
    # list the datasets of interest. This name should correspond to the
    # dataset name in the anndata object, 'dataset' column.
    # store the column name in the "cell type harmonization table" (google doc)
    # that corresponds to the dataset, in a dictionary:
    if ontology_type == "cell_type":
        harm_colnames = cell_type_harm_colnames
        # differ per dataset, so store that as dataset_cat
        dataset_cat = "dataset"
    elif ontology_type == "anatomical_region_coarse":
        last_authors = sorted(set(adata.obs["last_author/PI"]))
        harm_colnames = {la: la + "_coarse" for la in last_authors}
        # differ per last_author/PI, so store that as dataset_cat
        dataset_cat = "last_author/PI"
    elif ontology_type == "anatomical_region_fine":
        last_authors = sorted(set(adata.obs["last_author/PI"]))
        harm_colnames = {la: la + "_fine" for la in last_authors}
        # differ per last_author/PI, so store that as dataset_cat
        dataset_cat = "last_author/PI"
    else:
        raise ValueError(
            "ontology_type must be set to either cell_type, anatomical_region_coarse or anatomical_region_fine. Exiting"
        )
    # store set of datasets
    datasets = sorted(set(adata.obs[dataset_cat]))
    # harm_colnames["krasnow_ss2"] = "Krasnow"
    original_annotation_names = {
        "cell_type": "original_celltype_ann",
        "anatomical_region_coarse": "anatomical region coarse",
        "anatomical_region_fine": "anatomical region detailed",
    }
    original_annotation_name = original_annotation_names[ontology_type]
    original_annotations_prefixed = sorted(
        set(
            [
                cell_dataset + "_" + ann
                for cell_dataset, ann in zip(
                    adata.obs[dataset_cat], adata.obs[original_annotation_name]
                )
                if str(ann) != "nan"
            ]
        )
    )
    level_names = [
        "level" + "_" + str(level_number)
        for level_number in range(1, number_of_levels + 1)
    ]
    translation_df = pd.DataFrame(
        index=original_annotations_prefixed,
        columns=level_names + ["new", "highest_res"],
    )
    for dataset in datasets:
        if verbose:
            print("working on dataset " + dataset + "...")
        cell_type_harm_column = harm_colnames[dataset]
        # get cell type names, remove nan and None from list:
        cell_types_original = sorted(
            set(
                [
                    cell
                    for cell, cell_dataset in zip(
                        adata.obs[original_annotation_name], adata.obs[dataset_cat]
                    )
                    if cell_dataset == dataset
                    and str(cell) != "nan"
                    and str(cell) != "None"
                ]
            )
        )
        for label in cell_types_original:
            if verbose:
                print(label)
            # add dataset prefix for output, in that way we can distinguish
            # between identical labels in different datasets
            label_prefixed = dataset + "_" + label
            # if the label is 'nan', skip this row
            if label == "nan":
                continue
            # get the rows in which this label appears in the dataset column
            # of the harmonizing_df:
            ref_rows = harmonizing_df[cell_type_harm_column][
                harmonizing_df[cell_type_harm_column] == label
            ].index.tolist()
            # if the label does not occur, there is a discrepancy between the
            # labels in adata and the labels in the harmonizing table made
            # by Martijn and Sasha.
            if len(ref_rows) == 0:
                print(
                    "HEY THERE ARE NO ROWS IN THE harmonizing_df WITH THIS LABEL ({}) AS ENTRY!".format(
                        label
                    )
                )
            # if the label occurs twice, there is some ambiguity as to how to
            # translate this label to a consensus label. In that case, translate
            # to the finest level that is identical in consensus translation
            # between the multiple rows.
            elif len(ref_rows) > 1:
                print(
                    "HEY THE NUMBER OF ROWS WITH ENTRY {} IS NOT 1 but {}!".format(
                        label, str(len(ref_rows))
                    )
                )
                # get matching indices from consensus_df:
                consensus_idc = list()
                for i in range(len(ref_rows)):
                    consensus_idc.append(
                        consensus_df[
                            [
                                ref_rows[i] in index_list
                                for index_list in consensus_df["harmonizing_df_index"]
                            ]
                        ].index[0]
                    )
                # now get the translations for both cases:
                df_label_subset = consensus_df.loc[consensus_idc, :]
                # store only labels that are common among all rows with this label:
                for level in range(1, number_of_levels + 1):
                    col = "level_" + str(level)
                    n_translations = len(set(df_label_subset.loc[:, col]))
                    if n_translations == 1:
                        # update finest annotation
                        finest_annotation = df_label_subset.loc[consensus_idc[0], col]
                        # update finest annotation level
                        finest_level = level
                        # store current annotation at current level
                        translation_df.loc[label_prefixed, col] = finest_annotation

                    # if level labels differ between instances, store None for this level
                    else:
                        translation_df.loc[label_prefixed, col] = (
                            str(finest_level) + "_" + finest_annotation
                        )
                        translation_df.loc[label_prefixed, "highest_res"] = finest_level
                # set "new" to false, since we fell back to a common annotation:
                translation_df.loc[label_prefixed, "new"] = False
                # add ref rows to harmonizing_df_index?
                # ...
            # if there is only one row with this label, copy the consensus labels
            # from the same row to the translation df.
            else:
                consensus_idx = consensus_df[
                    [
                        ref_rows[0] in index_list
                        for index_list in consensus_df["harmonizing_df_index"]
                    ]
                ]
                consensus_idx = consensus_idx.index[
                    0
                ]  # loc[label,'harmonizing_df_index'][0]
                translation_df.loc[label_prefixed, :] = consensus_df.loc[
                    consensus_idx, :
                ]
    if verbose:
        print("Done!")
    return translation_df

def consensus_annotate_anndata(adata, translation_df, verbose=True, max_ann_level=5, ontology_type="cell_type"):
    """annotates cells in adata with consensus annotation. Returns adata."""
    # list the datasets of interest. This name should correspond to the
    # dataset name in the anndata object, 'dataset' column.
    if ontology_type == "cell_type":
        # cell types differ per dataset, so store that as dataset_cat
        dataset_cat = "dataset"
    elif ontology_type in ["anatomical_region_coarse", "anatomical_region_fine"]:
        # differ per last_author/PI, so store that as dataset_cat
        dataset_cat = "last_author/PI"
    else:
        raise ValueError(
            "ontology_type must be set to either cell_type, anatomical_region_coarse or anatomical_region_fine. Exiting"
        )
    # get name of adata.obs column with original annotations:
    original_annotation_names = {
        "cell_type": "original_celltype_ann",
        "anatomical_region_coarse": "anatomical region coarse",
        "anatomical_region_fine": "anatomical region detailed",
    }
    original_annotation_name = original_annotation_names[ontology_type]
    # add prefix to original annotations, so that we can distinguish between
    # datasets:
    adata.obs[original_annotation_name + "_prefixed"] = [
        cell_dataset + "_" + ann if str(ann) != "nan" else np.nan
        for cell_dataset, ann in zip(
            adata.obs[dataset_cat], adata.obs[original_annotation_name]
        )
    ]
    # specify the columns to copy:
    col_to_copy = [
        "level_" + str(level_number) for level_number in range(1, max_ann_level + 1)
    ]
    col_to_copy = col_to_copy + ["highest_res", "new"]
    # add prefix for in adata.obs:
    prefixes = {
        "cell_type": "ann_",
        "anatomical_region_coarse": "region_coarse_",
        "anatomical_region_fine": "region_fine_",
    }
    prefix = prefixes[ontology_type]
    col_to_copy_new_names = [prefix + name for name in col_to_copy]
    # add these columns to adata:
    for col_name_old, col_name_new in zip(col_to_copy, col_to_copy_new_names):
        translation_dict = dict(zip(translation_df.index, translation_df[col_name_old]))
        adata.obs[col_name_new] = adata.obs[original_annotation_name + "_prefixed"].map(
            translation_dict
        )
    adata.obs.drop([original_annotation_name + "_prefixed"], axis=1, inplace=True)
    return adata

# dictionary that translates dataset names to their matching column name
# in the cell type ontology csv (includes old and new dataset names)
# consider cleaning up later
cell_type_harm_colnames = {
    "Vanderbilt_Kropski_bioRxivHabermann": "Habermann_2019",
    "Sanger_Meyer_2019Madissoon": "Madissoon_2020",
    "Pittsburgh_Lafyatis_2019Morse": "Morse_2019",
    "Pittsburgh_Lafyatis_2019Morse_10Xv1": "Morse_2019",
    "Pittsburgh_Lafyatis_2019Morse_10Xv2": "Morse_2019",
    "Berlin_Eils_bioRxivLukassen": "Eils_2020",
    "Northwestern_Misharin_2018Reyfman": "Reyfman_2019",
    "BostonUniversity_Beane_2019Duclos": "covid_Duclos 2019",
    "BostonMedicalCenter_IDA": "covid_IDA",
    "BMC_SU2C_nasal": "covid_SU2C_nasal",
    "BMC?_SU2C_mixed": "covid_SU2C_mixed",
    "Koenigshoff": "covid_koenigshoff",
    "Sanger_Teichmann_2019VieiraBraga": "VieiraBraga_Sanger",
    "CNRS_Barbry_bioRxivDeprez": "Deprez_2020",
    "Misharin_new": "NU_CZI01",
    "Stanford_Krasnow_bioRxivTravaglini": "Krasnow",
    "Schiller": "covid_Schiller",
    "Xu_Whitsett_Peds_10X": "Whitsett_Xu",
    "Xu_Whitsett_Peds_dropseq": "Whitsett_Xu",
    "UMCG_Nawijn_2019VieiraBraga_lung": "Vieira-Braga_2019_F3_4",
    "UMCG_Nawijn_2019VieiraBraga_nasal": "VieiraBraga_UMCG_nasal",
    "NJH_Seibold_2020Goldfarbmuren": "Seibold_2020",
    "NJH_Seibold_2020Goldfarbmuren_10Xv2": "Seibold_2020",
    "NJH_Seibold_2020Goldfarbmuren_10Xv3": "Seibold_2020",
    "Vanderbilt_Kropski_bioRxivHabermann_vand": "Habermann_2019",
    "Vanderbilt_Kropski_bioRxivHabermann_dnar": "Habermann_2019",
    "Vanderbilt_Kropski_bioRxivHabermann_dna": "Habermann_2019",
    "Groningen_Nawijn_2019VieiraBraga_lung": "Vieira-Braga_2019_F3_4",
    "Sanger_Teichmann_2019VieiraBraga_UMCG_nasal": "VieiraBraga_UMCG_nasal",
    "Sanger_Teichmann_2019VieiraBraga_Sanger": "VieiraBraga_Sanger",
    "VieiraBraga_asthma": "Vieira-Braga_2019_F3_4",
    "regev_healthy_nucseq": "Regev",
    "tsankov": "Tsankov",
    "kaminski": "Kaminski",
    "schultze": "Schultze",
    "schultze_falk": "Schultze_Falk",
    "rawlins": "Rawlins_fetal",
    "xavier": "Xavier",
    "meyer_new": "Meyer_new",
    "spence_fetal": "Spence_fetal",
    "barbry": "Deprez_2020",
    "seibold": "Seibold",
    "shalek": "Shalek",
    "linnarsson": "Linnarsson_temp",
}


# ANATOMICAL REGION HARMONIZATION:

def merge_anatomical_annotations(ann_coarse, ann_fine):
    """Takes in two same-level annotation, for coarse and
    fine annotation, returns the finest annotation.
    To use on vectors, do:
        np.vectorize(merge_anatomical_annotations)(
        vector_coarse, vector_fine
    )
    """
    if utils.check_if_nan(ann_coarse):
        ann_coarse_annotated = False
    elif ann_coarse[0].isdigit() and ann_coarse[1] == "_":
        ann_coarse_annotated = ann_coarse[0]
    else:
        ann_coarse_annotated = True
    if utils.check_if_nan(ann_fine):
        ann_fine_annotated = False
    elif ann_fine[0].isdigit() and ann_fine[1] == "_":
        ann_fine_annotated = ann_fine[0]
    else:
        ann_fine_annotated = True
    # if finely annotated, return fine annotation
    if ann_fine_annotated == True:
        return ann_fine
    # if only coarse is annotated, return coarse annotation
    elif ann_coarse_annotated == True:
        return ann_coarse
    # if both are not annotated, return np.nan
    elif ann_coarse_annotated == False and ann_fine_annotated == False:
        return np.nan
    # if only one is not annotated, return the other:
    elif ann_coarse_annotated == False:
        return ann_fine
    elif ann_fine_annotated == False:
        return ann_coarse
    # if one or both are under-annotated (i.e. have
    # forward propagated annotations from higher levels),
    # choose the one with the highest prefix
    elif ann_coarse_annotated > ann_fine_annotated:
        return ann_coarse
    elif ann_fine_annotated > ann_coarse_annotated:
        return ann_fine
    elif ann_fine_annotated == ann_coarse_annotated:
        if ann_coarse == ann_fine:
            return ann_fine
        else:
            raise ValueError(
                "Contradicting annotations. ann_coarse: {}, ann_fine: {}".format(
                    ann_coarse, ann_fine
                )
            )
    else:
        raise ValueError(
            "Something most have gone wrong. ann_coarse: {}, ann_fine: {}".format(
                ann_coarse, ann_fine
            )
        )


def merge_coarse_and_fine_anatomical_ontology_anns(
    adata, remove_harm_coarse_and_fine_original=False, n_levels=3
):
    """takes in an adata with in its obs: anatomical_region_coarse_level_[n]
    and anatomical_region_fine_level_[n] and merges those for n_levels.
    Returns adata with merged annotation under anatomical_region_level_[n].
    Removes coarse and fine original harmonizations if 
    remove_harm_coarse_and_fine_original is set to True."""
    for lev in range(1, n_levels + 1):
        adata.obs["anatomical_region_level_" + str(lev)] = np.vectorize(
            merge_anatomical_annotations
        )(
            adata.obs["region_coarse_level_" + str(lev)],
            adata.obs["region_fine_level_" + str(lev)],
        )
    adata.obs["anatomical_region_highest_res"] = np.vectorize(max)(
        adata.obs["region_coarse_highest_res"], adata.obs["region_fine_highest_res"]
    )
    if remove_harm_coarse_and_fine_original:
        for lev in range(1, n_levels + 1):
            del adata.obs["region_coarse_level_" + str(lev)]
            del adata.obs["region_fine_level_" + str(lev)]
        del adata.obs["region_coarse_highest_res"]
        del adata.obs["region_fine_highest_res"]
        del adata.obs["region_coarse_new"]
        del adata.obs["region_fine_new"]
    return adata