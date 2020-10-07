from helper_functions.main_script_NDL import Generate_all_dictionary, Generate_corrupt_and_denoising_results
from helper_functions.final_plots_display import diplay_ROC_plots, all_dictionaries_display, top_dictionaries_display, all_dictionaries_display_rank

# output files will be saved in Network_dictionary/test


# Run main script to generate dictionary files and individual plots

Generate_all_dictionary()

# Generate Figures 2 and 7

top_dictionaries_display(motif_sizes=[6, 11, 21, 51, 101], latent_motif_rank=2)
top_dictionaries_display(motif_sizes=[6, 11, 21, 51, 101], latent_motif_rank=1)


# Generate Figures 8, 11

list_network_files = ['Caltech36.txt',
                      'MIT8.txt',
                      'UCLA26.txt',
                      'Harvard1.txt']

all_dictionaries_display(list_network_files, motif_sizes = [6, 11, 21, 51, 101] , name='1')
all_dictionaries_display_rank(list_network_files, name='1')


# Generate Figures 9, 12

list_network_files = ['COVID_PPI.txt',
                      'facebook_combined.txt',
                      'arxiv.txt',
                      'node2vec_homosapiens_PPI.txt']

all_dictionaries_display(list_network_files, motif_sizes = [6, 11, 21, 51, 101] , name='1')
all_dictionaries_display_rank(list_network_files, name='2')



# Generate Figures 10, 13

list_network_files = ['true_edgelist_for_ER_5000_mean_degree_50.txt',
                      'true_edgelist_for_ER_5000_mean_degree_100.txt',
                      'true_edgelist_for_SW_5000_k_50_p_0.05.txt',
                      'true_edgelist_for_SW_5000_k_50_p_0.1.txt',
                      'true_edgelist_for_BA_5000_m_25.txt',
                      'true_edgelist_for_BA_5000_m_50.txt']

all_dictionaries_display(list_network_files, motif_sizes = [6, 11, 21, 51, 101] , name='1')
all_dictionaries_display_rank(list_network_files, name='3')




# Run main script to generate ROC files for denoising experiments

Generate_corrupt_and_denoising_results()

# Properly modify the ROC file paths in helper_functions.final_plots_display.display_ROC_plots
# using the file names generated (all starts with "ROC_dict")

# Generate Figure 4 (latex table is separate)

diplay_ROC_plots()
