import os
from glob import glob
import yaml
import click

@click.command()
@click.option('-p', '--path-to-mlflow', help='Path to local folder with mlflow experiments')
@click.option('-id', '--exp-id', help='Experiment id in the specified mlflow folder to be modified.')
@click.option('-np', '--new-path-to-mlflow', help='New path to be set throughout meta.yaml configs for a specified mlflow experiment.')
@click.option('-nid', '--new-exp-id', default=None, help='If passed, will also reset the current experiment id.')
@click.option('-nn', '--new-exp-name', default=None, help='If passed, will also reset the current experiment name.')
def main(path_to_mlflow, exp_id, new_path_to_mlflow, new_exp_id, new_exp_name):
    path_to_exp = os.path.abspath(path_to_mlflow) + '/' + exp_id
    if not os.path.exists(path_to_exp):
        raise OSError(f'{path_to_exp} does not exist')
    if new_exp_id is not None:
        new_path_to_exp = os.path.abspath(new_path_to_mlflow) + '/' + new_exp_id
        renamed_path_to_exp = os.path.abspath(path_to_mlflow) + '/' + new_exp_id
        if os.path.exists(renamed_path_to_exp):
            print(f'\nFound an existing experiment with ID={new_exp_id} in {path_to_mlflow}. \
    If runs in this experiment have the same origin, please merge them into the currently processed experiment folder, remove original folder and rerun this script. \
    Otherwise, rename experiment {new_exp_id} to a new ID to avoid overlap: \
    \n\n    python {os.path.basename(__file__)} -p {path_to_mlflow} -id {new_exp_id} -np {path_to_mlflow} -nid NEW_EXP_ID_HERE\n') 
            return
    else:
        new_path_to_exp = os.path.abspath(new_path_to_mlflow) + '/' + exp_id
    
    meta_files = glob(f'{path_to_exp}/**/meta.yaml', recursive=True)
    run_folders = glob(f'{path_to_exp}/*/')
    main_meta_file = meta_files.pop(meta_files.index(f'{path_to_exp}/meta.yaml'))
    assert len(meta_files)==len(run_folders) # check that there is the same number of run folders as meta yamls (after popping the main one)

    # read main meta.yaml
    with open(main_meta_file, 'r') as f:
        exp_data = yaml.safe_load(f)
    
    # modify
    assert 'artifact_location' in exp_data
    exp_data['artifact_location'] = f'file://{new_path_to_exp}'
    if new_exp_id is not None:
        assert 'experiment_id' in exp_data
        exp_data['experiment_id'] = new_exp_id
    if new_exp_name is not None:
        assert 'name' in exp_data
        exp_data['name'] = new_exp_name
    
    # write
    with open(main_meta_file, 'w') as f:
        yaml.dump(exp_data, f)

    # loop over meta.yaml for each run 
    for meta_file in meta_files:
        # read run meta.yaml
        with open(meta_file, 'r') as f:
            run_data = yaml.safe_load(f)
        assert meta_file.split(run_data['run_id'])[-1] == '/meta.yaml' # check if this is a run-related meta.yaml
        
        # modify
        assert 'artifact_uri' in run_data
        run_data['artifact_uri'] = f"file://{new_path_to_exp}/{run_data['run_id']}/artifacts"
        if new_exp_id is not None:
            assert 'experiment_id' in run_data
            run_data['experiment_id'] = new_exp_id
        
        # write
        with open(meta_file, 'w') as f:
            yaml.dump(run_data, f)

    # rename local experiment folder to new id
    if new_exp_id is not None:
        os.rename(path_to_exp, renamed_path_to_exp)

if __name__ == '__main__':
    main()