import os
import umap
from umap.umap_ import UMAP
import torch
import pickle
import numpy as np
from tqdm import tqdm
import plotly.graph_objs as go
import plotly.io as pio
from datasets import load_dataset
from transformers import GPTJForCausalLM, AutoTokenizer


SUBSAMPLE_N = 10
SAVEDIR = './data'

def make_sphere_vis(model_strings):

    def create_scatter(x, y, z, text, name, mode, marker, line=None):
        if line is not None:
            scatter = go.Scatter3d(x=x, y=y, z=z, mode=mode, marker=marker, line=line, text=text, hoverinfo='text',
                                   name=name)
        else:
            scatter = go.Scatter3d(x=x, y=y, z=z, mode=mode, marker=marker, text=text, hoverinfo='text', name=name)
        return scatter

    with open('./data/figure_4_sphere_data.pkl', 'rb') as f:
        sphere_data_by_model_string = pickle.load(f)

    generations_by_model_string = {}
    for model_string in model_strings:
        save_string = model_string.replace('/', '_')
        with open('./data/' + save_string + '_fig4_generations.pkl', 'rb') as f:
            generations_by_model_string[model_string] = pickle.load(f)

    # Sample data for the first set of coordinates (replace with your own array of x, y, z coordinates and labels)
    tokenizer1 = AutoTokenizer.from_pretrained(model_strings[0])
    tokenizer2 = AutoTokenizer.from_pretrained(model_strings[1])
    for sample_idx in range(SUBSAMPLE_N):
        all_traces = []
        for i, coordinates1 in enumerate(sphere_data_by_model_string[model_strings[0]][sample_idx]):
            labels1 = tokenizer1.tokenize(generations_by_model_string[model_strings[0]][sample_idx][i])

            coordinates2 = sphere_data_by_model_string[model_strings[1]][sample_idx][i]
            labels2 = tokenizer2.tokenize(generations_by_model_string[model_strings[1]][sample_idx][i])

            k = 0
            for j, t in enumerate(labels1):
                if labels2[j] == t:
                    k +=1
                else:
                    break

            # Separate the coordinates into x, y, and z arrays for each set
            x1, y1, z1 = zip(*coordinates1)
            x2, y2, z2 = zip(*coordinates2)

            # Create a 3D scatter plot for each set of coordinates with stars for the first k points and dots for the rest
            if not i:
                scatter1_star = create_scatter(x1[:k], y1[:k], z1[:k], labels1[:k], 'GPTJ Prompt', 'lines+markers',
                                               marker=dict(size=8, symbol='cross', color='red', opacity=0.8))
                all_traces.append(scatter1_star)
                scatter2_star = create_scatter(x2[:k], y2[:k], z2[:k], labels2[:k], 'GPTJ-RLHF Prompt', 'lines+markers',
                                               marker=dict(size=8, symbol='cross', color='blue', opacity=0.8))

                all_traces.append(scatter2_star)

            scatter1_dot = create_scatter(x1[k:], y1[k:], z1[k:], labels1[k:], 'GPTJ Continuation '+str(i), 'lines+markers',
                                          marker=dict(size=6, symbol='circle', color='red', opacity=0.8))

            all_traces.append(scatter1_dot)
            scatter2_dot = create_scatter(x2[k:], y2[k:], z2[k:], labels2[k:], 'GPTJ-RLHF Continuation '+str(i), 'lines+markers',
                                          marker=dict(size=6, symbol='circle', color='blue', opacity=0.8))
            all_traces.append(scatter2_dot)

        # Define the layout
        layout = go.Layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                           margin=dict(l=0, r=0, b=0, t=0))

        print('at: ', all_traces)
        # Create a figure and add the scatter plots
        fig = go.Figure(data=all_traces, layout=layout)

        # Save the figure as an interactive HTML file
        pio.write_html(fig, file='./figs/3d_scatter_plot_sampled_labels_{}.html'.format(sample_idx))

def make_sphere_data(model_strings):

    if os.path.exists('./data/figure_4_sphere_data.pkl'):
        print('Found cached sphere data')
        return

    generations_by_model_string = {}
    pre_umap = []
    for model_string in model_strings:
        save_string = model_string.replace('/', '_')

        with open('./data/' + save_string + '_fig4_embeddings.pkl', 'rb') as f:
            all_embs = pickle.load(f)
            for prompt in all_embs:
                for sequence in prompt:
                    sequence = np.squeeze(sequence)
                    sequence = np.stack(sequence).astype('float64')
                    sequence = sequence / np.linalg.norm(sequence, axis=1, keepdims=True)
                    sequence = np.cumsum(sequence, axis=0) / np.arange(1, sequence.shape[0]+1).reshape(-1, 1)
                    pre_umap.append(np.stack(sequence))
            generations_by_model_string[model_string] = all_embs

    pre_umap = np.concatenate(pre_umap, axis=0)
    # sphere_mapper = umap.UMAP(output_metric='haversine').fit(pre_umap)
    sphere_mapper = UMAP(output_metric='haversine').fit(pre_umap)
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])

    pointer = 0
    sphere_data_by_model_string = {model_string: [] for model_string in model_strings}
    for model_string in model_strings:
        for prompt in generations_by_model_string[model_string]:
            cur_prompt_continuations = []
            for sequence in prompt:
                cur_generation_points = []
                for _ in sequence:
                    cur_generation_points.append([x[pointer], y[pointer], z[pointer]])
                    pointer += 1
                cur_prompt_continuations.append(cur_generation_points)
            sphere_data_by_model_string[model_string].append(cur_prompt_continuations)
    with open('./data/figure_4_sphere_data.pkl', 'wb') as f:
        pickle.dump(sphere_data_by_model_string, f)




def get_data(model_string):
    save_string = model_string.replace('/', '_')
    if os.path.exists('./data/' + save_string + '_fig4_embeddings.pkl'):
        print('found cached embeddings for model: ', save_string)
        with open('./data/' + save_string + '_fig4_embeddings.pkl', 'rb') as f:
            d = pickle.load(f)
            return [e for e in d[:10]]

    print('loading model')
    model = GPTJForCausalLM.from_pretrained(model_string,
                                            output_hidden_states=True,
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True)
    model.cuda()
    print('loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    print('loading dataset')
    ds = load_dataset('allenai/prosocial-dialog', split='test')
    print('formatting dataset')

    chosen_ds = [e['context'] for e in ds][:SUBSAMPLE_N]
    all_embeddings = []
    all_generations = []
    for e in tqdm(chosen_ds):
        input = tokenizer(e, return_tensors='pt', truncation=True, max_length=256)
        input = {k: v.cuda() for k, v in input.items()}


        n_new_tokens = 20
        cur_embeddings = []
        cur_generations = []
        for i in range(5):
            generated_tokens = model.generate(input['input_ids'],
                                              attention_mask=input['attention_mask'],
                                              max_length=input['input_ids'].shape[1] + n_new_tokens,
                                              min_new_tokens=10,
                                              num_beams=5,
                                              repetition_penalty=2.0,
                                              do_sample=True,
                                              num_return_sequences=1)


            generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            cur_generations.append(generated_text)

            # Concatenate the new tokens to the original input_ids
            extended_input_ids = torch.cat((input['input_ids'], generated_tokens[:, -n_new_tokens:]), dim=1)
            extended_attention_mask = torch.cat(
                (input["attention_mask"], torch.ones_like(generated_tokens[:, -n_new_tokens:]).cuda()), dim=1)

            extended_input = {"input_ids": extended_input_ids, "attention_mask": extended_attention_mask}

            y = model(**extended_input)
            embedding = y.hidden_states[0].detach().cpu().numpy()[0]
            cur_embeddings.append(embedding)
            cur_generations.append(generated_text)

        cur_embeddings = np.stack(cur_embeddings)
        all_embeddings.append(cur_embeddings)
        all_generations.append(cur_generations)


    with open('./data/' + save_string + '_fig4_embeddings.pkl', 'wb') as f:
        pickle.dump(all_embeddings, f)
    with open('./data/' + save_string + '_fig4_generations.pkl', 'wb') as f:
        pickle.dump(all_generations, f)

    model.cpu()
    del model

if __name__ == '__main__':
    _ = get_data("EleutherAI/gpt-j-6B")
    _ = get_data("reciprocate/ppo_hh_gpt-j")
    make_sphere_data(["EleutherAI/gpt-j-6B", "reciprocate/ppo_hh_gpt-j"])
    make_sphere_vis(["EleutherAI/gpt-j-6B", "reciprocate/ppo_hh_gpt-j"])