<p align="center">
<img width="600" src="https://github.com/ykureh/NMF-Networks/blob/master/NDL_logo.png?raw=true" alt="logo">
</p>


# NMF-Networks

This is the script used to learn a dictionary and code given a network.
------------------------------------------------

1) network_reconstruction_nx.py

To run:
./network_cross_reconstruction_nx.py college_index_of_school_to_learn number_of_entries_in_dict

For example, suppose you want to learn a 9 entry dictionary and code from Caltech36's network, then run:

./network_cross_reconstruction_nx.py 36 9




This is the script used to reconstruct a network given a dictionary.
------------------------------------------------

2) network_cross_reconstruction_nx.py

To run:
./network_cross_reconstruction_nx.py college_index_of_school_to_reconstruct college_index_of_dictionary_to_use number_of_entries_in_dict

For example, suppose you've learned Harvard1's dictionary with 9 entries, and now you want to reconstruct Caltech36's network from it, then run:

./network_cross_reconstruction_nx.py 36 1 9

