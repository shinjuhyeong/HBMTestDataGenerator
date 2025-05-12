import os
import numpy as np  # Ensure numpy is imported for P_TABLE_CU calculation
import subprocess
import json
# Store the initial current directory at the start of the script

def initialize_directories():
    # Use the stored initial directory to avoid issues with changing working directories
    current_dirc = CURRENT_DIRC
    print(f"Current Directory: {current_dirc}")

    # Create SIMU_DIRC directory
    SIMU_DIRC = os.path.join(current_dirc, 'SIMU_DIRC')
    if not os.path.exists(SIMU_DIRC):
        os.makedirs(SIMU_DIRC)
        print(f"Created folder: {SIMU_DIRC}")
    else:
        for root, dirs, files in os.walk(SIMU_DIRC, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
                print(f"Deleted folder: {dir_path}")
        print(f"Cleared all files and subdirectories in folder: {SIMU_DIRC}")

    # Create TRAINING_DATASET directory
    TRAINING_DATASET_DIR = os.path.join(current_dirc, 'TRAINING_DATASET')
    if not os.path.exists(TRAINING_DATASET_DIR):
        os.makedirs(TRAINING_DATASET_DIR)
        print(f"Created folder: {TRAINING_DATASET_DIR}")


    # Create IMAGES directory
    IMAGES_DIR = os.path.join(TRAINING_DATASET_DIR, 'IMAGES')
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
        print(f"Created folder: {IMAGES_DIR}")
    
    return current_dirc

def create_info_file(working_dirc, info_dict, file_name):
    infofile_path = os.path.join(working_dirc, file_name)
    with open(infofile_path, 'w') as f:
        json.dump(info_dict, f)
    return infofile_path



def create_part_images(working_dirc, geometrical_info, scaler=1):
    from scipy.interpolate import lagrange
    import numpy as np
    from PIL import Image
    # Unpack geometrical information
    R_SI = geometrical_info['R_SI']
    Y_SI = geometrical_info['Y_SI']
    R0_CU = geometrical_info['R0_CU']
    Y0_CU = geometrical_info['Y0_CU']

    R_SI_SCALED = R_SI * scaler
    Y_SI_SCALED = Y_SI * scaler
    R0_CU_SCALED = [r * scaler for r in geometrical_info['R0_CU']]
    Y0_CU_SCALED = [y * scaler for y in geometrical_info['Y0_CU']]

    # r-z relation
    r_CU = lagrange(Y0_CU_SCALED, R0_CU_SCALED)

    # Image dimensions
    width = int(R_SI_SCALED)  # x-axis spans from 0 to R_SI
    height = int(Y_SI_SCALED)      # y-axis spans from 0 to Y_SI

    # Create a blank image (white background)
    image = Image.new('RGB', (width, height), 'white')
    pixels = image.load()

    # Iterate over each pixel and determine its color based on r_CU
    for y in range(height):
        r_value = r_CU(y)
        for x in range(width):
            if x < r_value:  # Left of r_CU
                pixels[x, y] = (255, 0, 0)  # Red
            else:  # Right of r_CU
                pixels[x, y] = (128, 128, 128)  # Gray

    # Save the image in the working directory
    image_title = f"R_SI_{R_SI}_Y_SI_{Y_SI}_R0_CU_{'_'.join(map(str, R0_CU))}_Y0_CU_{'_'.join(map(str, Y0_CU))}.bmp"
    image_path = os.path.join(working_dirc, image_title)
    image.save(image_path)
    print(f"Image saved at: {image_path}")
    return image_path
    
def couple_crackresult_to_images(working_dirc, image_dirc, simu_dirc, scaler, csv_filename = 'results.csv'):
    # Check if the CSV file exists, if not, create an empty one
    csv_path = os.path.join(working_dirc, csv_filename)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as csv_file:
            csv_file.write('')  # Create an empty CSV file
        print(f"Created empty CSV file: {csv_path}")

    # Read the failed elements from the text file
    failed_elements_path = os.path.join(simu_dirc, 'failed_elements.txt')
    # Read the first element from the file
    with open(failed_elements_path, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0:
            first_line = lines[0].strip()
            # Extract the time and instance name from the first line
            crack_time, crack_instance_name, crack_element_label = first_line.split(', ')
        else:
            print("No lines found in the failed elements file.")
            return False

    # Find centroid coordinates from the text file
    initial_coordinates_path = os.path.join(simu_dirc, 'initial_coordinates.txt')
    with open(initial_coordinates_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            instance_name, element_label, centroidx, centroidy, centroidz = line.split(', ')
            if instance_name == crack_instance_name and element_label == crack_element_label:
                centroidx = float(centroidx)
                centroidy = float(centroidy)   
                centroidz = float(centroidz)
                # Calculate the coordinates
                centroidr = ((centroidx**2)+(centroidz**2))**(0.5)
                r_coord = (centroidr * scaler)
                y_coord = (centroidy * scaler)

                # Open the CSV file in append mode ('a') to avoid overwriting
                with open(csv_path, 'a') as csv_file:
                    print(f"{image_dirc}, {crack_time}, {r_coord}, {y_coord}\n")
                    csv_file.write(f"{image_dirc}, {crack_time}, {r_coord}, {y_coord}\n")

CURRENT_DIRC = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
SIMU_DIRC = os.path.join(CURRENT_DIRC, 'SIMU_DIRC')
TRAINING_DATASET_DIRC = os.path.join(CURRENT_DIRC, 'TRAINING_DATASET')
IMAGES_DIRC = os.path.join(TRAINING_DATASET_DIRC, 'IMAGES')

path_dict = {
    'CURRENT_DIRC': CURRENT_DIRC,
    'SIMU_DIRC': SIMU_DIRC,
    'TRAINING_DATASET_DIRC': TRAINING_DATASET_DIRC,
    'IMAGES_DIRC': IMAGES_DIRC
}
#check if the info file already exists, and if so, delete it
if os.path.exists('path_info.json'):
    os.remove('path_info.json')

create_info_file(CURRENT_DIRC, path_dict, 'path_info.json')

for r1 in range(10, 20, 2):
    for r2 in range(r1, r1 + 5):
        for r3 in range(r2 - 5, r2 + 5, 2):
            for y2 in range(30, 90, 10):
                initialize_directories()

                geometry_info = {
                    'R_SI': 100,
                    'Y_SI': 100,
                    'R0_CU': [r1, r2, r3],
                    'Y0_CU': [0, y2, 100]
                }

                material_properties = {
                    'E_CU': 120e3,  # Elastic Modulus
                    'NU_CU': 0.34,  # Poisson's Ratio
                    'CTE_CU': 17e-6,  # Coefficient of Thermal Expansion
                    'Y_CU': 150,  # Yield Strength
                    'P_TABLE_CU': [(140 + 69.6 * (e_t_cu ** 0.286), e_t_cu) for e_t_cu in np.linspace(0, 0.2, 200, endpoint=True).tolist()],
                    'K_CU': 0.401,  # Thermal Conductivity
                    'C_CU': 390e-6,  # Specific Heat
                    'RHO_CU': 8.960e-12,  # Density
                    'DAMAGEINIT_TABLE_CU': ((0.15, 0.1, 0.0),),
                    'DAMAGEEVOL_TABLE_CU': ((0.1,),),
                    'E_SI': 130000,  # Elastic Modulus
                    'NU_SI': 0.28,  # Poisson's Ratio
                    'CTE_SI': 2.8e-6,  # Coefficient of Thermal Expansion
                    'K_SI': 0.149,  # Thermal Conductivity
                    'C_SI': 700e-6,  # Specific Heat
                    'RHO_SI': 2.33e-12  # Density
                }

                thermal_cycle_info = {
                    'single_cycle_data': [
                        (0.0, 25.0),    # 초기 온도 25도
                        (1.833, 300.0),    # 15도/분으로 가열하여 18.33분 후 300°C 도달
                        (2.833, 300.0),    # 10분간 300°C 유지
                        (4.433, -65.0),    # -10도/분으로 냉각하여 44.33분 후 -65°C 도달
                        (5.433, -65.0),    # 10분간 -65°C 유지
                        (7.00, 25.0)       # 15도/분으로 가열하여 70분 후 25°C 도달
                    ],
                    'total_cycles': 10
                }
                #save geometry_info, material_properties, thermal_cycle_info in seperate txt file
                create_info_file(SIMU_DIRC, geometry_info, 'geometry_info.json')
                create_info_file(SIMU_DIRC, material_properties, 'material_properties.json')
                create_info_file(SIMU_DIRC, thermal_cycle_info, 'thermal_cycle_info.json')

                try:
                    # Run run_simulation.py
                    simul_script = 'abaqus cae noGUI=run_simulation.py'
                    print(f"Running simulation script: {simul_script}")
                    subprocess.run(simul_script, shell=True)
                    
                    report_script = 'abaqus python create_report.py'
                    print(f"Running simulation script: {report_script}")
                    subprocess.run(report_script, shell=True)

                    # Create images
                    scaler = 10
                    imagefile_dirc = create_part_images(IMAGES_DIRC, geometry_info, scaler)
                    couple_crackresult_to_images(TRAINING_DATASET_DIRC, imagefile_dirc, SIMU_DIRC, scaler)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
