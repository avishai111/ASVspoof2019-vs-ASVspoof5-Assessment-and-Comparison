import sklearn  
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os

def plotting_genuine(umap_train, embedded_groups, chosen_labels_sex, chosen_labels_logical_mapping, 
                            plot_title='Train embeddings (Only Bonafide)', gender='both', frontsize=20):
    """
    Plot UMAP embeddings for genuine data filtered by gender.

    Parameters:
        umap_train: Fitted UMAP model.
        embedded_groups: NumPy array of embeddings.
        chosen_labels_sex: Array indicating the gender ('male' or 'female') for each embedding.
        chosen_labels_logical_mapping: Array indicating if an embedding is genuine (e.g., #0000FF for genuine).
        plot_title: Title of the plot (default: 'Train embeddings (Only Genuine)').
        gender: Gender filter for the plot ('male', 'female', or 'both').
        frontsize: Font size for the plot labels and ticks.

    Returns:
        combined_embeddings: NumPy array of combined embeddings for the selected gender(s).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter genuine embeddings based on logical mapping
    genuine_filter = chosen_labels_logical_mapping == "#0000FF"  # Assuming #0000FF indicates genuine

    if gender == 'male':
        embeddings = umap_train.transform(
            embedded_groups[(chosen_labels_sex == 'male') & genuine_filter]
        )
        ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c='blue',  # Use a single color for genuine
            alpha=0.5, label='Male (Bonafide)'
        )
        combined_embeddings = embeddings

    elif gender == 'female':
        embeddings = umap_train.transform(
            embedded_groups[(chosen_labels_sex == 'female') & genuine_filter]
        )
        ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c='green',  # Use a single color for genuine
            alpha=0.5, label='Female (Bonafide)'
        )
        combined_embeddings = embeddings

    elif gender == 'both':
        male_embeddings = umap_train.transform(
            embedded_groups[(chosen_labels_sex == 'male') & genuine_filter]
        )
        female_embeddings = umap_train.transform(
            embedded_groups[(chosen_labels_sex == 'female') & genuine_filter]
        )
        ax.scatter(
            male_embeddings[:, 0], male_embeddings[:, 1],
            c='blue',  # Use a single color for genuine
            alpha=0.5, label='Male (Bonafide)'
        )
        ax.scatter(
            female_embeddings[:, 0], female_embeddings[:, 1],
            c='green',  # Use a single color for genuine
            alpha=0.5, label='Female (Bonafide)'
        )
        combined_embeddings = np.vstack((male_embeddings, female_embeddings))

    # Legend setup for genders
    ax.legend(fontsize=frontsize, loc='lower left')
    ax.set_title(plot_title, fontsize=24)
    ax.set_xlabel('UMAP 1', fontsize=24)
    ax.set_ylabel('UMAP 2', fontsize=24)
    ax.set_ylim(-10, 15)
    ax.set_xlim(0, 20)
    plt.xticks(fontsize=20)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=20)  # Increase font size for y-axis ticks
    plt.show()

    return combined_embeddings





def plotting_genuine_by_without_codec(umap_train, embedded_groups, chosen_labels_sex, chosen_labels_logical_mapping, 
                              chosen_labels_codec, codec_list, plot_title='Train embeddings (Only Genuine by Codec)', 
                              gender='both', frontsize=20):
    """
    Plot UMAP embeddings for genuine data filtered by gender and codec type, with each codec in a separate subplot.
    If the codec is '-', plot all embeddings in a single plot.

    Parameters:
        umap_train: Fitted UMAP model.
        embedded_groups: NumPy array of embeddings.
        chosen_labels_sex: Array indicating the gender ('male' or 'female') for each embedding.
        chosen_labels_logical_mapping: Array indicating if an embedding is genuine (e.g., #0000FF for genuine).
        chosen_labels_codec: Array indicating the codec type for each embedding.
        codec_list: List of codec types to filter and plot.
        plot_title: Title of the plot (default: 'Train embeddings (Only Genuine by Codec)').
        gender: Gender filter for the plot ('male', 'female', or 'both').
        frontsize: Font size for the plot labels and ticks.

    Returns:
        combined_embeddings: NumPy array of combined embeddings for the selected gender(s) and codec(s).
    """
    # Filter genuine embeddings based on logical mapping
    genuine_filter = chosen_labels_logical_mapping == "#0000FF"  # Assuming #0000FF indicates genuine

    if '-' in codec_list:
        fig, ax = plt.subplots(figsize=(10, 8))
        combined_embeddings = []

        for codec in codec_list:
            if codec == '-':
                codec_filter = chosen_labels_codec == codec

                if gender == 'both':
                    male_embeddings = umap_train.transform(
                    embedded_groups[(chosen_labels_sex == 'male') & genuine_filter & codec_filter]
                    )
                    female_embeddings = umap_train.transform(
                        embedded_groups[(chosen_labels_sex == 'female') & genuine_filter & codec_filter]
                    )
                    ax.scatter(
                        male_embeddings[:, 0], male_embeddings[:, 1],
                        c='blue',  # Use a single color for genuine
                        alpha=0.5, label=f'Male (Bonafide, No Codec)'
                    )
                    ax.scatter(
                        female_embeddings[:, 0], female_embeddings[:, 1],
                        c='green',  # Use a single color for genuine
                        alpha=0.5, label=f'Female (Bonafide, No Codec)'
                    )
                    embeddings = np.vstack((male_embeddings,female_embeddings))
                combined_embeddings.append(embeddings)

        combined_embeddings = np.vstack(combined_embeddings) if combined_embeddings else None

        # Set plot title and labels
        ax.set_title(plot_title, fontsize=24)
        ax.set_xlabel('UMAP 1', fontsize=24)
        ax.set_ylabel('UMAP 2', fontsize=24)
        ax.set_ylim(-10, 15)
        ax.set_xlim(0, 20)
        ax.legend(fontsize=frontsize, loc='lower left')
        plt.xticks(fontsize=20)  # Increase font size for x-axis ticks
        plt.yticks(fontsize=20)  # Increase font size for y-axis ticks
        plt.tight_layout()
        plt.show()

        return combined_embeddings
    
    

def plotting_genuine_by_all_codec(umap_train, embedded_groups, chosen_labels_sex, chosen_labels_logical_mapping, 
                              chosen_labels_codec, codec_list, plot_title='Train embeddings (Only Genuine by Codec)', 
                              gender='both', frontsize=20):
    """
    Plot UMAP embeddings for genuine data filtered by gender and codec type, with each codec in a separate subplot.
    If the codec is '-', plot all embeddings in a single plot.

    Parameters:
        umap_train: Fitted UMAP model.
        embedded_groups: NumPy array of embeddings.
        chosen_labels_sex: Array indicating the gender ('male' or 'female') for each embedding.
        chosen_labels_logical_mapping: Array indicating if an embedding is genuine (e.g., #0000FF for genuine).
        chosen_labels_codec: Array indicating the codec type for each embedding.
        codec_list: List of codec types to filter and plot.
        plot_title: Title of the plot (default: 'Train embeddings (Only Genuine by Codec)').
        gender: Gender filter for the plot ('male', 'female', or 'both').
        frontsize: Font size for the plot labels and ticks.

    Returns:
        combined_embeddings: NumPy array of combined embeddings for the selected gender(s) and codec(s).
    """
    # Filter genuine embeddings based on logical mapping
    genuine_filter = chosen_labels_logical_mapping == "#0000FF"  # Assuming #0000FF indicates genuine

    if '-' in codec_list:
        fig, ax = plt.subplots(figsize=(10, 8))
        combined_embeddings = []

        for codec in codec_list:
            if codec != '-':
                codec_filter = chosen_labels_codec == codec

                if gender == 'both':
                    male_embeddings = umap_train.transform(
                    embedded_groups[(chosen_labels_sex == 'male') & genuine_filter & codec_filter]
                    )
                    female_embeddings = umap_train.transform(
                        embedded_groups[(chosen_labels_sex == 'female') & genuine_filter & codec_filter]
                    )
                    ax.scatter(
                        male_embeddings[:, 0], male_embeddings[:, 1],
                        c='blue',  # Use a single color for genuine
                        alpha=0.5, label=f'Male (Bonafide, With Codec)'
                    )
                    ax.scatter(
                        female_embeddings[:, 0], female_embeddings[:, 1],
                        c='green',  # Use a single color for genuine
                        alpha=0.5, label=f'Female (Bonafide, With Codec)'
                    )
                    embeddings = np.vstack((male_embeddings,female_embeddings))
                combined_embeddings.append(embeddings)

        combined_embeddings = np.vstack(combined_embeddings) if combined_embeddings else None

        # Set plot title and labels
        ax.set_title(plot_title, fontsize=24)
        ax.set_xlabel('UMAP 1', fontsize=24)
        ax.set_ylabel('UMAP 2', fontsize=24)
        ax.set_ylim(-10, 15)
        ax.set_xlim(0, 20)
        ax.legend(fontsize=frontsize, loc='lower left')
        plt.xticks(fontsize=20)  # Increase font size for x-axis ticks
        plt.yticks(fontsize=20)  # Increase font size for y-axis ticks
        plt.tight_layout()
        plt.show()

        return combined_embeddings
    
    

