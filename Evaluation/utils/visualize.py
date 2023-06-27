import tensorflow as tf
from hydra import initialize, compose

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import widgets

from utils.training import compose_datasets
from models.transformer import Transformer


class Event:
    # ---------------------------------------------------------------------------------------
    # Object which allows to visualise certain aspects about events within a batch of inputs.
    # ---------------------------------------------------------------------------------------

    def __init__(self):
        # ~~~~~~~ Configurations: ~~~~~~~
        mpl.rcParams['figure.dpi'] = 300

        with initialize(config_path='../configs/'):
            self.cfg = compose(config_name='train.yaml')
        assert self.cfg['model']['kwargs']['encoder']['embedding_kwargs']['r_cut'] is None, \
                        'r_cut is not None -> please align sequence preprocessing in _get_attn_data' \
                                              'with those done inside model.call()'

        self.feature_name_to_idx = {}
        for particle_type, names in self.cfg['feature_names'].items():
            self.feature_name_to_idx[particle_type] = {name: i for i,name in enumerate(names)}

        self.TYPE_MAP = {
            0: 'Undefined',
            1: 'Charged Hadron',
            2: 'Electron',
            3: 'Muon',
            4: 'Photon',
            5: 'Neutral Hadron',
            6: 'HF Tower as Hadron',
            7: 'HF Tower as EM Particle'
        }

        N_INNER_CELLS   = 11
        INNER_CELL_SIZE = 0.02
        N_OUTER_CELLS   = 21
        OUTER_CELL_SIZE = 0.05

        self.INNER_LOC = N_INNER_CELLS*INNER_CELL_SIZE/2
        self.OUTER_LOC = N_OUTER_CELLS*OUTER_CELL_SIZE/2

        self.text_size = 18

        # ~~~~~~~ Dataset: ~~~~~~~
        self.train_data, self.val_data = compose_datasets(self.cfg['datasets'], self.cfg['tf_dataset_cfg'])

        # ~~~~~~~ Models: ~~~~~~~
        self.trained_model = tf.keras.models.load_model('./trained_models/transformer_model/artifacts/model/')
        self.model = Transformer(self.feature_name_to_idx, self.cfg['model']['kwargs']['encoder'], self.cfg['model']['kwargs']['decoder'])

        # ~~~~~~~ Retrieving Particles and Attention Data: ~~~~~~~
        self.sequence_data, self.attn_data = self._get_attn_data(self.model, self.trained_model, self.train_data, self.val_data)
        self.particles = [self._create_particle_df(example_num=i) for i in range(self.cfg['tf_dataset_cfg']['train_batch_size'])]


    # Visualising Attention Functions
    ################################################################################################################################

    def show_attn_matrices(self, layer, include_global=False, example_num=0):
        # ---------------------------------------------------------------------------
        # Function show the attention matrices of every head within a specified layer
        # ---------------------------------------------------------------------------

        # Extracts matrices of required layer
        scores = [self.attn_data[l][example_num,:,:,:] for l in range(len(self.attn_data))]

        # Removes last row and column (ie. the global feature token) of each matrix if needed
        scores = scores[layer] if include_global else scores[layer][:,:-1,:-1]

        with plt.style.context('seaborn-pastel'):
            fig, ax = plt.subplots(2,4, sharex=True, sharey=True, figsize=(32,12), constrained_layout=True)

            for i in range(2):
                for j in range(4):
                    ax[i][j].matshow(scores[i*4 + j])
                    ax[i][j].set_title(f'Attention Head: {i*4+j + 1}', size=self.text_size)


    def compute_attn_weighted_X(self, X, layer, plot=False, bins=20, example_num=0):
        # ------------------------------------------------------------------------------------------------
        # Functions calculates an attention weighted average of distances/momenta/global scores*.
        # It can also, for each head seperately, plot all the score-distance pairs.
        # 
        # * For global scores it just returns the average score of the last column (global token column)
        # ------------------------------------------------------------------------------------------------

        assert X == 'distance' or X == 'pt' or X == 'global' or X == 'sd_wo_global' or X == 'sd_all'
        assert bins >= 0
        assert 0 <= example_num <= self.cfg['tf_dataset_cfg']['train_batch_size']
        assert 0 <= layer <= self.cfg['model']['kwargs']['encoder']['num_layers']-1

        if X == 'distance':
            # Gets a dataframe with all the data
            attn_vs_X_dfs = self._get_attn_X_data(X='distance', layer=layer, bins=bins, example_num=example_num)
        elif X == 'pt':
            # Gets a dataframe with all the data
            attn_vs_X_dfs = self._get_attn_X_data(X='pt', layer=layer, bins=bins, example_num=example_num)
        elif X == 'global':
            # Gets last column of attention matrix, then averages across that column
            global_scores = self.attn_data[layer][example_num,:,:,-1:]
            global_scores = tf.squeeze(global_scores)
            global_scores = tf.math.reduce_mean(global_scores, axis=-1)

            return global_scores.numpy()
        elif X == 'sd_wo_global':
            # Gets all attention scores, except from last row and last column, then computes standard deviation
            scores = self.attn_data[layer][example_num,:,:-1,:-1]
            sds = [tf.math.reduce_std(scores[i]).numpy() for i in range(8)]

            return sds
        elif X == 'sd_all':
            # Gets all attention scores then computes standard deviation
            scores = self.attn_data[layer][example_num,:,:,:]
            sds = [tf.math.reduce_std(scores[i]).numpy() for i in range(8)]

            return sds

        if plot:
            with plt.style.context('utils/style.mplstyle'): # Custom style sheet
                fig, ax = plt.subplots(2,4, sharex=True, sharey=True, figsize=(32,12), constrained_layout=True)

                for i in range(2):
                    for j in range(4):
                        ax[i][j].scatter(
                            attn_vs_X_dfs[i*4+j].index, 
                            attn_vs_X_dfs[i*4+j]['score'], 
                            s=50, 
                            facecolor='#A0C4FF',
                            edgecolor='black'
                        )
                        ax[i][j].set_title(f'Attention Head: {i*4+j + 1}', size=self.text_size)

                        if X == 'pt': ax[i][j].set_xscale('log')
                    
                if X == 'distance': 
                    xlabel = 'Distance between Particles (bins)'
                elif X == 'pt':
                    xlabel = 'Average Momentum between Particles (bins)'

                fig.supxlabel(xlabel, size=self.text_size)
                fig.supylabel('Mean Attention Score', size=self.text_size)

        weighted_averages = []
        for head in range(8):
            # Computes weighted average. For accurate weighted average, bins should be set to 0.
            X_ = attn_vs_X_dfs[head].index.to_numpy()
            weight = attn_vs_X_dfs[head]['score'].to_numpy()
            weight = np.nan_to_num(weight)

            average = np.dot(X_, weight) / np.sum(weight)
            weighted_averages.append(average)
        
        return weighted_averages


    def show_meanAttnX_vs_layer(self, X, plot=False, example_num=0, output=False):
        # ---------------------------------------------------------------------------------------------------------------
        # Functions uses the attention weighted X, where X=distance/momentum/global score, of each head and plots them
        # against the layer, creating a ViT inspired graph.
        # ---------------------------------------------------------------------------------------------------------------

        assert X == 'distance' or X == 'pt' or X == 'global' or X == 'sd_wo_global' or X == 'sd_all'
        assert 0 <= example_num <= self.cfg['tf_dataset_cfg']['train_batch_size']

        # Lists that will be used for the dataframe creation.
        averages_per_layer = []
        layers = []
        heads = []
        for layer in range(6):
            avgs = self.compute_attn_weighted_X(X=X, layer=layer, bins=0, example_num=example_num)
            averages_per_layer.append(avgs)

            for head in range(8):
                layers.append(layer)
                heads.append(head)

        # Creates dataframe with: layer number, head number, and distance/momentum/etc.
        df = pd.DataFrame({
            'layer': layers,
            'head': heads,
            f'{X}': np.array(averages_per_layer).flatten()
        })

        if plot:
            with plt.style.context('utils/style.mplstyle'):
                fig, ax = plt.subplots(1,1,figsize=(8,6),constrained_layout=True)

                # Plotting heads as new ax.scatter()
                for head in range(8):
                    if head < 3: label = f'Head {head + 1}'
                    elif head == 3: label = '...'
                    else: label = None

                    ax.scatter(
                        df.groupby('head').get_group(head)['layer'], # Retrives rows of current head, then looks at layer number data
                        df.groupby('head').get_group(head)[f'{X}'], # Retrieves rows of current head, then looks at mean distance/pt/etc..
                        edgecolor='black',
                        s=100,
                        label=label
                    )

                if X == 'distance': 
                    ylabel = 'Mean Distance'
                elif X == 'pt':
                    ylabel = 'Mean Momentum'
                elif X == 'global':
                    ylabel = 'Mean Global Score'
                elif X == 'sd_wo_global':
                    ylabel = 'Standard Deviation of all Scores w/o Global Scores'
                elif X == 'sd_all':
                    ylabel = 'Standard Deviation of all Scores'

                ax.set_xlabel('Network depth (layer)', size=self.text_size)
                ax.set_ylabel(ylabel, size=self.text_size)
                ax.legend()

        # For plotting purposes, output is not required. For an average of multiple events, output is needed.
        if output: 
            return df

    
    def show_multiple_meanAttnX_vs_layer(self, X, num_events=20):
        # ---------------------------------------------------------------------------------------------------------------
        # Functions uses the attention weighted X, where X=distance/momentum/global/standard deviation, of each head and plots them
        # against the layer, creating a ViT inspired graph averaged from multiple events.
        # ---------------------------------------------------------------------------------------------------------------

        assert X == 'distance' or X == 'pt' or X == 'global' or X == 'sd_wo_global' or X == 'sd_all'
        assert num_events <= self.cfg['tf_dataset_cfg']['train_batch_size']

        data = []
        for i in range(num_events):
            # Getting all the data
            # print(f'{i+1}/{num_events}')
            data.append(self.show_meanAttnX_vs_layer(X=X, plot=False, example_num=i, output=True))

        # Creating one dataframes out of single event dataframes.
        df = pd.concat(data, axis=0)

        with plt.style.context('utils/style.mplstyle'):
            fig, ax = plt.subplots(1,6,figsize=(8,6),constrained_layout=True, sharey=True)

            for layer in range(6):
                curr_layer_data = df.groupby('layer').get_group(layer) # Retrieves current layer data
                values = []

                for head in range(8):
                    # Gets head data from each head seperately, then appends to a list. Done because of the way boxplots wants the data (list of lists).
                    curr_head_data = curr_layer_data.groupby('head').get_group(head)
                    curr_head_data = curr_head_data[f'{X}'].to_numpy()

                    values.append(curr_head_data)

                mean = np.mean(values)

                ax[layer].boxplot(
                    values,
                    patch_artist=True, 
                    boxprops=dict(facecolor='lightskyblue', color='black'),
                    capprops=dict(color='black'),
                    whiskerprops=dict(color='black'),
                    flierprops=dict(color='black', markeredgecolor='black'),
                    medianprops=dict(color='black'),
                )
                ax[layer].plot(
                    [1,8],
                    [mean, mean],
                    color='red'
                )
                ax[layer].set_title(f'Layer {layer+1}', size=self.text_size)
                
            if X == 'distance': 
                ylabel = 'Mean Distance'
            elif X == 'pt':
                ylabel = 'Mean Momentum'
            elif X == 'global':
                ylabel = 'Mean Global Score'
            elif X == 'sd_wo_global':
                ylabel = 'Standard Deviation of all Scores w/o Global Scores'
            elif X == 'sd_all':
                ylabel = 'Standard Deviation of all Scores'

            fig.supylabel(ylabel, size=self.text_size)
            fig.supxlabel('Head', size=self.text_size)


    # Widget:
    ################################################################################################################################

    def show_widget(self, example_num=0):
        # ----------------------------------------------------------------------------- 
        # Widget showing particles of an event and attention scores in deta-dphi space.        
        # ----------------------------------------------------------------------------- 

        # Creating initial figure
        fig = self._create_figure(layer=0, head='Combined', example_num=example_num)

        # Adding cones showing inner and outer range of DeepTau
        fig.add_shape(
            type='circle',
            xref='x', yref='y',
            x0=-self.INNER_LOC, y0=-self.INNER_LOC, x1=self.INNER_LOC, y1=self.INNER_LOC,
            line_color='RoyalBlue'
        )
        fig.add_shape(
            type='circle',
            xref='x', yref='y',
            x0=-self.OUTER_LOC, y0=-self.OUTER_LOC, x1=self.OUTER_LOC, y1=self.OUTER_LOC,
            line_color='LightCoral'
        )

        # Fine tuning the layout
        fig.update_layout(
            autosize=False, 
            width=650, 
            height=650,
            margin=dict(l=50, r=150, t=50, b=150)
        )
        fig.update_xaxes(range=[-1.2, 1.2])
        fig.update_yaxes(range=[-1.2, 1.2])

        # Creating the widget
        g = go.FigureWidget(data=fig, layout=go.Layout(title=dict(text='Widget')))

        # Adding sliders, etc...
        layer_slider = widgets.IntSlider(
            value=0, 
            min=0, 
            max=5, 
            step=1, 
            description='Layer', 
            continuous_update=False
        )

        head_options = [f'Head {i+1}' for i in range(8)]
        head_options.append('Combined')
        head_box = widgets.Dropdown(
            value='Combined',
            options=head_options,  
            description='Attention Head'
        )

        threshold_text = widgets.FloatText(
            value=0.1,
            description='Attn. Score Threshold'
        )

        showWeights_box = widgets.Checkbox(
            value=True,
            description='Show Weights'
        )

        container1 = widgets.HBox(children=[showWeights_box, layer_slider])
        container2 = widgets.HBox(children=[head_box, threshold_text])
            
        # Response function checking for changes in values of sliders, etc.
        def response(change):  
            layer = layer_slider.value
            threshold = threshold_text.value
            head = head_box.value
            showWeights = showWeights_box.value
            
            fig = self._create_figure(layer, head, threshold, showWeights, example_num=example_num)
                
            with g.batch_update():
                g.data = ()
                [g.add_trace(item) for item in fig.data]
                    
                g.layout.title.text = f'Layer = {layer}, Head = {head}, Threshold = {threshold}'
            
        layer_slider.observe(response, names="value")
        threshold_text.observe(response, names="value")
        head_box.observe(response, names="value")
        showWeights_box.observe(response, names="value")

        return container1, container2, g


    def _create_figure(self, layer, head, threshold=0.1, show_weights=True, example_num=0):
        particles = self.particles[example_num]

        fig = px.scatter(particles, x="deta", y="dphi", color="type", 
                     size='pt', hover_data=['pt', 'deta', 'dphi'],
                     title=f'Layer = {layer}, Head = {head}, Threshold = {threshold}')

        if show_weights:
            particles = particles.to_numpy()
            particle_scores = [
                self.attn_data[l][example_num, :, :particles.shape[0], :particles.shape[0]] for l in range(len(self.attn_data))
            ]
            
            # Retrieving correct attention matrix based on head setting
            if head == 'Combined': scores = tf.math.reduce_mean(particle_scores[layer], axis=0)
            elif head == 'Head 1': scores = particle_scores[layer][0,:,:]
            elif head == 'Head 2': scores = particle_scores[layer][1,:,:]
            elif head == 'Head 3': scores = particle_scores[layer][2,:,:]
            elif head == 'Head 4': scores = particle_scores[layer][3,:,:]
            elif head == 'Head 5': scores = particle_scores[layer][4,:,:]
            elif head == 'Head 6': scores = particle_scores[layer][5,:,:]
            elif head == 'Head 7': scores = particle_scores[layer][6,:,:]
            elif head == 'Head 8': scores = particle_scores[layer][7,:,:]
        
            # Looping through all cells in attention matrix
            for i,row in enumerate(scores):
                for j,elem in enumerate(row):
                
                    # Too many lines to plot w/o some threshold
                    if elem > threshold:
                        alpha = -np.exp(-4*(elem - threshold+0.001)) + 1 # Somewhat arbitrary mapping, used to make differences in scores more visible
                        
                        fig.add_trace(go.Scatter(
                            x = [particles[i][0], particles[j][0]],
                            y = [particles[i][1], particles[j][1]],
                            mode = 'lines', line = dict(color=f'rgba(150,150,150,{alpha})') 
                        ))
  
        return fig

    
    # Private Functions of Class:
    ################################################################################################################################

    def _get_attn_data(self, model, trained_model, train_data, val_data, seq_len_cap=None):
        # ---------------------------------------------------------------------------------------------------------------
        # Function initialises a model with pre-trained weights, then passes one batch through and retrives corresponding
        # attention scores.
        # ---------------------------------------------------------------------------------------------------------------

        # To initialise weights
        for X,y in train_data:
            outputs, _ = model(X)
            break

        # Setting weights to weights of trained model
        for ind,layer in enumerate(model.layers):
            layer.set_weights(trained_model.layers[ind].get_weights()) 

        # Get attention scores for input batch
        for X,y in val_data:
            # Skipping every sequence that is too long
            if seq_len_cap != None and X[0].to_tensor().shape[1] >= seq_len_cap: continue

            # Extracting a particles
            sequence = [
                X[0].to_tensor(),
                X[1].to_tensor(),
                X[2].to_tensor(),
                X[3]
            ]

            # Retrieving attention scores
            _, attn_scores = model(X) # attn_scores is list of attention scores per layer: attn_scores[layer][:batch_size, :num_heads, :num_tokens, :num_tokens]

            break

        return sequence, attn_scores


    def _create_particle_df(self, example_num=0):
        # ----------------------------------------------------------------------------------------------------
        # Function creates dataframes from tensor data. 
        # Dataframe includes: deta, dphi, pt, type
        # ----------------------------------------------------------------------------------------------------

        particles = [self.sequence_data[i][example_num] for i in range(len(self.sequence_data))]
        
        # Getting coordinates in terms of deta and dphi for each type
        deta_pfCand = particles[0][:,self.feature_name_to_idx['pfCand']['r']] * np.cos(particles[0][:,self.feature_name_to_idx['pfCand']['theta']])
        dphi_pfCand = particles[0][:,self.feature_name_to_idx['pfCand']['r']] * np.sin(particles[0][:,self.feature_name_to_idx['pfCand']['theta']])

        deta_ele = particles[1][:,self.feature_name_to_idx['ele']['r']] * np.cos(particles[1][:,self.feature_name_to_idx['ele']['theta']])
        dphi_ele = particles[1][:,self.feature_name_to_idx['ele']['r']] * np.sin(particles[1][:,self.feature_name_to_idx['ele']['theta']])
        
        deta_muon = particles[2][:,self.feature_name_to_idx['muon']['r']] * np.cos(particles[2][:,self.feature_name_to_idx['muon']['theta']])
        dphi_muon = particles[2][:,self.feature_name_to_idx['muon']['r']] * np.sin(particles[2][:,self.feature_name_to_idx['muon']['theta']])

        deta = [*deta_pfCand.numpy(), *deta_ele.numpy(), *deta_muon.numpy()]
        dphi = [*dphi_pfCand.numpy(), *dphi_ele.numpy(), *dphi_muon.numpy()]

        # Making a list of the different particles types
        types_pfCand = [self.TYPE_MAP[x+1] for x in particles[0][:,self.feature_name_to_idx['pfCand']['particle_type']].numpy()]
        type_ele = ['ELECTRON' for i in range(len(particles[1]))]
        type_muon = ['MUON' for i in range(len(particles[2]))]

        particle_type = [*types_pfCand, *type_ele, *type_muon]
        
        # List of all pts
        pt_pfCand = np.abs(particles[0][:,self.feature_name_to_idx['pfCand']['rel_pt']])
        pt_ele = np.abs(particles[1][:,self.feature_name_to_idx['ele']['rel_pt']])
        pt_muon = np.abs(particles[2][:,self.feature_name_to_idx['muon']['rel_pt']])

        pt = [*pt_pfCand, *pt_ele, *pt_muon]

        # Creating the data frame
        tau_df = pd.DataFrame({
            'deta': deta, 
            'dphi': dphi,
            'pt': pt, 
            'type': particle_type 
        })

        return tau_df


    def _get_attn_X_data(self, X, layer, bins=20, example_num=0):
        # ----------------------------------------------------------------------------------------------------------------------
        # Function which creates a dataframe, for each head, containing the distances/average momenta between particles and the 
        # corresponding score. So, for each cell in an attention matrix the distance/average momenutm between the particles and 
        # the score. It also puts the distances/average momenta into bins, and averages across the bin.
        # If bins = 0, then data from all cells is returned.
        # ----------------------------------------------------------------------------------------------------------------------

        assert X == 'distance' or X == 'pt'
        assert bins >= 0
        assert 0 <= example_num <= self.cfg['tf_dataset_cfg']['train_batch_size']
        assert 0 <= layer <= self.cfg['model']['kwargs']['encoder']['num_layers']-1

        attn_scores = self.attn_data[layer][example_num,:,:-1,:-1]
        particles = self.particles[example_num].to_numpy()

        dataframes = []
        for head in range(8):
            X_values = []
            scores = []
            
            for i,row in enumerate(attn_scores[head,:,:]):
                for j,score in enumerate(row):
                    if X == 'distance':
                        # Calculating distance between the corresponding particles of a cell.
                        val = np.sqrt(
                            (particles[i][0] - particles[j][0])**2 + (particles[i][1] - particles[j][1])**2
                        )
                    elif X == 'pt':
                        # Calculating the average pt of the 2 particles corresponding to a paticular cell.
                        val = (particles[i][self.feature_name_to_idx['pfCand']['rel_pt']] + particles[j][self.feature_name_to_idx['pfCand']['rel_pt']]) / 2

                    X_values.append(val)
                    scores.append(score.numpy())
                
            df = pd.DataFrame({'X': X_values, 'score': scores})

            if bins == 0:
                # Setting distance as index, needed for later functions
                df = df.set_index('X')
            else:
                # Creates bins
                n_bins = np.linspace(0, max(X_values), bins)

                # Sorts into bins, and averages within bin
                df['bucket'] = pd.cut(df['X'], bins)
                df['bucket'] = df['bucket'].apply(
                    lambda x: (x.left + x.right) / 2 
                )
                df = df.groupby(['bucket'], group_keys=False).mean()

            dataframes.append(df)

        return dataframes
