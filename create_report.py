from odbAccess import openOdb
import os
import json

def fetch_info_from_json(working_dirc, filename):
    file_dirc = os.path.join(working_dirc, filename)
    with open(file_dirc, 'r') as f:
        info_dict = json.load(f)
    return info_dict


def fetch_odb_path(working_dirc):
    """
    Fetch the ODB file path from the working directory.

    Parameters:
        working_dirc (str): Path to the working directory.

    Returns:
        str: Path to the ODB file.
    """
    # List all files in the working directory
    files = os.listdir(working_dirc)
    
    # Filter for ODB files
    odb_files = [f for f in files if f.endswith('.odb')]
    
    if len(odb_files) == 0:
        raise FileNotFoundError("No ODB file found in the specified directory.")
    
    # Return the first ODB file found
    return os.path.join(working_dirc, odb_files[0])

def save_initial_coordinates(odb_path, output_file):


    """
    Save the initial coordinates of all elements in the ODB file, averaged to (x, y, z) format.

    Parameters:
        odb_path (str): Path to the ODB file.
        output_file (str): Path to the output text file.
    """
    # Open the ODB file
    print(f"Opening ODB file: {odb_path}")
    odb = openOdb(path=odb_path)

    with open(output_file, 'w') as f:
        #f.write("Initial Element Coordinates\n")
        #f.write("================================\n")

        for instance_name, instance in odb.rootAssembly.instances.items():
            for element in instance.elements:
                element_label = element.label
                nodes = element.connectivity
                node_coords = [list(instance.nodes[node - 1].coordinates) for node in nodes]
                coordsum_x = 0.0
                coordsum_y = 0.0
                coordsum_z = 0.0
                # Calculate centroid as the average of node coordinates
                for coord in node_coords:
                    coordsum_x = coord[0] + coordsum_x
                    coordsum_y = coord[1] + coordsum_y
                    coordsum_z = coord[2] + coordsum_z
                centroidx = coordsum_x / len(node_coords)
                centroidy = coordsum_y / len(node_coords)  
                centroidz = coordsum_z / len(node_coords)
                centroid = [centroidx, centroidy, centroidz]    

                f.write(f"{instance_name}, {element_label}, {centroidx}, {centroidy}, {centroidz} \n")

    odb.close()
    print("Finished processing ODB file.")

def extract_failed_elements(odb_path, output_file):

    """
    Extracts failed elements and their corresponding step time from an Abaqus ODB file.
    Records only the first failure of each element.

    Parameters:
        odb_path (str): Path to the ODB file.
        output_file (str): Path to the output text file.
    """
    # Delete the lock file if it exists
    lock_file = odb_path.replace('.odb', '.lck')
    if os.path.exists(lock_file):
        os.remove(lock_file)

    # Open the ODB file
    odb = openOdb(path=odb_path)

    with open(output_file, 'w') as f:
        #f.write("Failed Elements and Step Times\n")
        #f.write("================================\n")

        # Set to track already recorded failed elements
        recorded_elements = set()

        # Iterate through steps
        for step_name, step in odb.steps.items():
            #f.write(f"Step: {step_name}\n")

            # Iterate through frames in the step
            for frame in step.frames:
                time = frame.frameValue  # Step time
                status_field = frame.fieldOutputs['STATUS']  # Use direct key access instead of 'get'

                for value in status_field.values:
                    if value.data == 0:  # Failed element
                        element_label = value.elementLabel
                        instance_name = value.instance.name
                        element_key = (instance_name, element_label)

                        # Record only if the element is not already recorded
                        if element_key not in recorded_elements:
                            recorded_elements.add(element_key)
                            f.write(f"{time}, {instance_name}, {element_label}\n")

    odb.close()


CURRENT_DIRC = os.path.dirname(os.path.abspath('create_report.py'))
path_dict = fetch_info_from_json(CURRENT_DIRC, 'path_info.json')
SIMU_DIRC = path_dict['SIMU_DIRC']

output_odb_dirc = fetch_odb_path(SIMU_DIRC)
save_initial_coordinates(output_odb_dirc, os.path.join(SIMU_DIRC, 'initial_coordinates.txt'))
extract_failed_elements(output_odb_dirc, os.path.join(SIMU_DIRC, 'failed_elements.txt'))
