from allensdk.core.reference_space_cache import ReferenceSpaceCache


reference_space_key = 'annotation/ccf_2017'
resolution = 25
rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1)


def get_all_children_id_by_struct_id(id):
    ancestor_id_map = tree.get_ancestor_id_map()
    all_children_id_list = []
    for key, value in ancestor_id_map.items():
        if id in value:
            all_children_id_list.append(key)
    return all_children_id_list


def get_allchildren_id_by_struct_name(name):
    ancestor_id_map = tree.get_ancestor_id_map()
    all_children_id_list = []
    id = get_struct_id_by_name(name)
    for key, value in ancestor_id_map.items():
        if id in value:
            all_children_id_list.append(key)
    return all_children_id_list


def get_all_ancestor_id_by_struct_id(id):
    region = tree.get_structures_by_id([id])[0]
    struct_id_path = region['structure_id_path']
    ancestor_list = []
    for i in range(len(struct_id_path)-1):
        ancestor_list.append(struct_id_path[i])
    return ancestor_list


def get_struct_id_by_name(name):
    """
    get the struct id using full name of structure
    """
    region = tree.get_structures_by_name([name])[0]
    print(region)
    assert region is not None
    return region['id']


def get_color_by_struct_id(id):
    region = tree.get_structures_by_id([id])[0]
    print(region)
    assert region is not None
    return region['rgb_triplet']


def get_color_by_struct_name(name):
    sid = get_struct_id_by_name(name)
    return get_color_by_struct_id(sid)


def get_struct_name_by_struct_id(id):
    region = tree.get_structures_by_id([id])[0]
    print(region)
    assert region is not None
    return region['name']
