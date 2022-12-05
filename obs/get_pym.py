#%%
"""
python==3.10.0
mp_api==0.25.2
"""

from mp_api.client import MPRester
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
api_key = "PvfnzQv5PLh4Lzxz1pScKnAtcmmWVaeU" #"___your MP API key___"
generate_fig = False

#%%
def get_struct(idx=1000):
    """_summary_

    Args:
        idx (int): mp-id. Defaults to 1000.

    Returns:
        structure: pymatgen
    """
    mpid = 'mp-' + str(idx)
    with MPRester(api_key) as mpr:
        struct = mpr.get_structure_by_material_id(mpid)

    return struct

#%%
if __name__=="__main__":
    struct = get_struct(idx=1000)

#%%

# %%
