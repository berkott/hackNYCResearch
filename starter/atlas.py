from nomic import atlas
from datasets import load_dataset

SUBSAMPLE_N = 1000

print('loading dataset')
ds = load_dataset('allenai/prosocial-dialog', split='test')
print('formatting dataset')

documents = [{
        "text": ds[i]['context'], 
        "label": ds[i]['safety_label'], 
        'url': f'https://berkott.github.io/hackNYCResearch/starter/figs/3d_scatter_plot_sampled_labels_{i}.html', 
        "id": str(i)
    } for i in range(SUBSAMPLE_N)]

project = atlas.map_text(data=documents,
                          indexed_field='text',
                          name='Mysterious Graphs',
                          id_field='id',
                          colorable_fields=['label'],
                          description='Distance is just by text similarity, not path similarity!'
                          )
