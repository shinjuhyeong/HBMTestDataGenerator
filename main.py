import os
import numpy as np  # Ensure numpy is imported for P_TABLE_CU calculation
import subprocess
import json
import logging
from datetime import datetime
import random

# Store the initial current directory at the start of the script
def initialize_directories(makelog=True):
    current_dirc = CURRENT_DIRC
    if makelog:
        logging.info(f"Current Directory: {current_dirc}")

    # Create SIMU_DIRC directory
    simu_dirc = os.path.join(current_dirc, 'SIMU_DIRC')
    if not os.path.exists(simu_dirc):
        os.makedirs(simu_dirc)
        if makelog:
            logging.info(f"Created folder: {simu_dirc}")
    else:
        for root, dirs, files in os.walk(simu_dirc, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        if makelog:
            logging.info(f"Cleared all files and subdirectories in folder: {simu_dirc}")

    # Create TRAINING_DATASET_DIRC directory
    training_dataset_dirc = os.path.join(current_dirc, 'TRAINING_DATASET')
    if not os.path.exists(training_dataset_dirc):
        os.makedirs(training_dataset_dirc)
        if makelog:
            logging.info(f"Created folder: {training_dataset_dirc}")

    # Create IMAGES_DIRC directory
    images_dirc = os.path.join(training_dataset_dirc, 'IMAGES')
    if not os.path.exists(images_dirc):
        os.makedirs(images_dirc)
        if makelog:
            logging.info(f"Created folder: {images_dirc}")
           
    return current_dirc


#create_info_file : create a json file in the working directory based on the info_dict
#Input: working_dirc(type: str), info_dict(type: dict), file_name(type: str)
#Output: infofile_path(type: str)
def create_info_json(working_dirc, info_dict, file_name):
    infofile_dict = os.path.join(working_dirc, file_name)
    if os.path.exists(infofile_dict):
        os.delete(infofile_dict)
    with open(infofile_dict, 'w') as f:
        json.dump(info_dict, f)
    return infofile_dict


#create_image : create an image based on the geometrical information
#Input: images_dirc(type: str), geometrical_info(type: dict), scaler(type: float)
#Output: imagefile_dirc(type: str), image_size(type: list)
def create_image(images_dirc, geometrical_info, scaler=1, makelog=True):
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
                pixels[x, height - (y + 1)] = (255, 0, 0)  # Red
            else:  # Right of r_CU
                pixels[x, height - (y + 1)] = (128, 128, 128)  # Gray

    # Save the image in the working directory
    image_title = f"R_SI_{R_SI}_Y_SI_{Y_SI}_R0_CU_{'_'.join(map(str, R0_CU))}_Y0_CU_{'_'.join(map(str, Y0_CU))}.bmp"
    imagefile_dirc = os.path.join(images_dirc, image_title)
    #image_path_relative starts from 상위폴더 of csv_file
    image.save(imagefile_dirc)
    if makelog:
        logging.info(f"Image saved at: {imagefile_dirc}")

    image_size = [width, height]
    return imagefile_dirc, image_size


# create_coupled_traindata : create a csv file with the image and crack data
# Input: training_dataset_dirc(type: str), images_dirc(type: str), geometrical_info(type: dict), crack_data(type: list), cracknum(type: int), scaler(type: float)
# Output: imagefile_dirc(type: str)
def create_coupled_traindata(training_dataset_dirc, images_dirc, geometrical_info, crack_data, cracknum, scaler=1, csv_filename='results.csv', makelog = True): 
    if crack_data == []:
        if makelog:
            logging.info("No cracks found in the simulation.")
        return False

    csvfile_dirc = os.path.join(training_dataset_dirc, csv_filename)
    if not os.path.exists(csvfile_dirc):
        with open(csvfile_dirc, 'w') as csv_file:
            csv_file.write('')  # Create an empty CSV file
        if makelog:
            logging.info(f"Created empty CSV file: {csvfile_dirc}")

    # If coupled_crackdata has more than cracknum, save the first cracknum cracks
    if len(crack_data) > cracknum:
        coupled_crackdata_trunc = crack_data[:cracknum]
    else:
        coupled_crackdata_trunc = crack_data
    imagefile_dirc, size = create_image(images_dirc, geometrical_info, scaler)
    image_path_relative = os.path.relpath(imagefile_dirc, os.path.dirname(csvfile_dirc))
    with open(csvfile_dirc, 'a') as csv_file:
        for crack in coupled_crackdata_trunc:
            csv_file.write(f"{image_path_relative}, ")
            crack_time, r, y = crack
            # Save the results to the CSV file, with respect to the image file
            r_coord = r * scaler
            y_coord = size[1] - (y * scaler + 1)
            csv_file.write(f"{crack_time}, {r_coord}, {y_coord}, ")
            csv_file.write("\n")
            if makelog:
                logging.info(f"Crack data saved: {crack_time}, {r_coord}, {y_coord}")

    return imagefile_dirc


# crackdata_from_reports : read the report file and extract the crack data
# Input: simu_dirc(type: str)
# Output: coupled_crackdata(type: list)    
def crackdata_from_reports(simu_dirc, makelog=True):
    # return coupled_crackdata = list of lists <= coupled with scaler
    # Check if the CSV file exists, if not, create an empty one
    coupled_crackdata = []
    # Read the failed elements from the text file
    failed_elements_path = os.path.join(simu_dirc, 'failed_elements.txt')
    initial_coordinates_path = os.path.join(simu_dirc, 'initial_coordinates.txt')
    # Read the first element from the file
    if not os.path.exists(failed_elements_path):
        if makelog:
            logging.info(f"Failed elements file not found in dirc: {failed_elements_path}")
        return coupled_crackdata
    if not os.path.exists(initial_coordinates_path):
        if makelog:
            logging.info(f"Initial coordinates file not found in dirc: {initial_coordinates_path}")
        return coupled_crackdata
    with open(failed_elements_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            crack_time, crack_instance_name, crack_element_label = line.split(', ')
            #print(f"Crack time: {crack_time}, Crack instance name: {crack_instance_name}, Crack element label: {crack_element_label}")
            # Find centroid coordinates from the text file
            with open(initial_coordinates_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    instance_name, element_label, centroidx, centroidy, centroidz = line.split(', ')
                    #print(f"Instance name: {instance_name}, Element label: {element_label}, Centroid coordinates: {centroidx}, {centroidy}, {centroidz}")
                    if instance_name == crack_instance_name and int(element_label) == int(crack_element_label):
                        centroidx = float(centroidx)
                        centroidy = float(centroidy)   
                        centroidz = float(centroidz)
                        # Calculate the coordinates
                        centroidr = ((centroidx**2)+(centroidz**2))**(0.5)
                        crack_time = float(crack_time)
                        # Append the coordinates to the list
                        coupled_crackdata.append([crack_time, centroidr, centroidy])
                        if makelog:
                            logging.info(f"time, centroid_(r,y): {crack_time}, {centroidr}, {centroidy}")
    return coupled_crackdata


#fetch_completed_conditions_from_csv : read the results.csv file and fetch the completed conditions
#Input: resultsfile_dirc(type: str)
#resultsfile should be in the format of 'R0_CU_10_12_14_Y0_CU_30_50_100'
#Output: completed_simulation_list(type: list)
#completed_simulation_list = [[R0_CU, Y0_CU], ...], where R0_CU and Y0_CU are lists of integers
def fetch_completed_conditions_from_csv(resultsfile_dirc):
    completed_simulation_list = []
    if os.path.exists(resultsfile_dirc):
        with open(resultsfile_dirc, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parsed_list = line.split(', ')
                fetched_imagefiledirc = parsed_list[0]
                
                #parse image_dirc to get three numbers after R0_CU and YO_CU each
                fetched_imagefiledirc = fetched_imagefiledirc.split('_')
                for i in range(len(fetched_imagefiledirc)):
                    if fetched_imagefiledirc[i].startswith('R0'):
                        r0_cu = fetched_imagefiledirc[i+2:i+5]
                        r0_cu = [int(x) for x in r0_cu]
                    elif fetched_imagefiledirc[i].startswith('Y0'):
                        y0_cu = fetched_imagefiledirc[i+2:i+5]
                        #delete .bmp if it exists in the last element of y0_cu
                        y0_cu[-1] = y0_cu[-1].split('.')[0]
                        y0_cu = [int(x) for x in y0_cu]
                completed_simulation_list.append([r0_cu, y0_cu])
    return completed_simulation_list


#Preliminary setup for paths
#--------------------------------------------------------------#
CURRENT_DIRC = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
SIMU_DIRC = os.path.join(CURRENT_DIRC, 'SIMU_DIRC')
TRAINING_DATASET_DIRC = os.path.join(CURRENT_DIRC, 'TRAINING_DATASET')
IMAGES_DIRC = os.path.join(TRAINING_DATASET_DIRC, 'IMAGES')
LOG_DIRC = os.path.join(CURRENT_DIRC, 'logs')

if not os.path.exists(LOG_DIRC):
    os.makedirs(LOG_DIRC)
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S")
log_filename = os.path.join(LOG_DIRC, f'hbmtestdatagenerator-{formatted_date}.log')
logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.info(f"Log file generated at {formatted_date}")

path_dict = {
    'CURRENT_DIRC': CURRENT_DIRC,
    'SIMU_DIRC': SIMU_DIRC,
    'TRAINING_DATASET_DIRC': TRAINING_DATASET_DIRC,
    'IMAGES_DIRC': IMAGES_DIRC
}
#check if the info file already exists, and if so, delete it
if os.path.exists('path_info.json'):
    os.remove('path_info.json')

create_info_json(CURRENT_DIRC, path_dict, 'path_info.json')

# ASK USER FOR REMOVAL OF TRAINING_DATASET_DIRC
if os.path.exists(TRAINING_DATASET_DIRC):
    print(f"Do you want to remove {TRAINING_DATASET_DIRC} and its contents? (y/n)")
    user_input = input()
    logging.info(f"1st user input for asking removal: {user_input}")
    if user_input.lower() == 'y':
        print(f"Are you sure you want to remove {TRAINING_DATASET_DIRC} and its contents? (y/n)")
        user_input = input()
        logging.info(f"2nd user input for asking removal: {user_input}")
        if user_input.lower() == 'y':
            for root, dirs, files in os.walk(TRAINING_DATASET_DIRC, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(TRAINING_DATASET_DIRC)
            logging.info(f"Deleted folder: {TRAINING_DATASET_DIRC}")
#---------------------------------------------------------------#
#make experiment_conditions

simulation_conditions = []
for r1 in range(10, 20, 2):
    for r2 in range(r1, r1 + 5):
        for r3 in range(r2 - 5, r2 + 5, 2):
            for y2 in range(30, 60, 10):
                if abs(r1 - r2) <= 2 and abs(r2 - r3) <= 2: #안나옴
                    continue
                else:               
                    simulation_conditions.append([r1, r2, r3, y2])
random.shuffle(simulation_conditions)

total_cycles_conditions = [25, 50]

for total_cycles in total_cycles_conditions:
    initialize_directories()
    #read result.csv file and check completed simulation, save to completed_simulation_list
    resultsfile_dirc = os.path.join(TRAINING_DATASET_DIRC, 'results.csv')
    completed_conditions = fetch_completed_conditions_from_csv(resultsfile_dirc)
    failed_conditions_for_same_y = []
    
    for condition in simulation_conditions:  
        r1, r2, r3, y2 = condition
        #실패한 y와 같은 y에 대해서는 실패할 가능성이 높음 : wiseflag
        
        logging.info(f"Try a simulation for R0_CU: {r1}, {r2}, {r3} and Y0_CU: {y2}, total_cycles: {total_cycles}")

        complete_flag = False
        failed_for_same_y_flag = False

        for completed_condition in completed_conditions:
            if [r1, r2, r3] == completed_condition[0] and y2 == completed_condition[1][1]:
                logging.info(f"Simulation already completed for R0_CU: {r1}, {r2}, {r3} and Y0_CU: {y2}")
                complete_flag = True
                break

        if [r1, r2, r3] in failed_conditions_for_same_y:
            logging.info(f"Simulation previously failed for R0_CU but different y2, Skip for current cycle number: {r1}, {r2}, {r3}")
            failed_for_same_y_flag = True

        if complete_flag or failed_for_same_y_flag:
            continue
        
        initialize_directories()

        geometrical_info = {
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
            'total_cycles': total_cycles
        }
        
        mesh_info = {
            'mesh_size' : 10
        }

        #save geometry_info, material_properties, thermal_cycle_info in seperate txt file
        create_info_json(SIMU_DIRC, geometrical_info, 'geometrical_info.json')
        create_info_json(SIMU_DIRC, material_properties, 'material_properties.json')
        create_info_json(SIMU_DIRC, thermal_cycle_info, 'thermal_cycle_info.json')
        create_info_json(SIMU_DIRC, mesh_info, 'mesh_info.json')

        try:
            simul_script = 'abaqus cae noGUI=run_simulation.py'
            logging.info(f"Running simulation script: {simul_script}")
            subprocess.run(simul_script, shell=True)
            
            learning_error = False
            outputrec_dirc = os.path.join(SIMU_DIRC, "auto_model_output.rec")
            if os.path.exists(outputrec_dirc):
                with open(outputrec_dirc, 'r') as log_file:
                    for line in log_file:
                        if "Learning" in line:
                            learning_error = True

            if learning_error == True:
                logging.info("Learning Edition Error has occurred, Try for a bigger node")        
                initialize_directories()

                geometrical_info = {
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
                    'total_cycles': total_cycles
                }
                
                mesh_info = {
                    'mesh_size' : 15
                }

                #save geometrical_info, material_properties, thermal_cycle_info in seperate txt file
                create_info_json(SIMU_DIRC, geometrical_info, 'geometrical_info.json')
                create_info_json(SIMU_DIRC, material_properties, 'material_properties.json')
                create_info_json(SIMU_DIRC, thermal_cycle_info, 'thermal_cycle_info.json')
                create_info_json(SIMU_DIRC, mesh_info, 'mesh_info.json')

                subprocess.run(simul_script, shell=True)

            
            report_script = 'abaqus python create_report.py'
            logging.info(f"Running simulation script: {report_script}")
            subprocess.run(report_script, shell=True)
            
            # Create images
            scaler = 10
            resulted_crack_data = crackdata_from_reports(SIMU_DIRC, scaler)
            if resulted_crack_data == []:
                if [r1, r2, r3] not in failed_conditions_for_same_y:
                    failed_conditions_for_same_y.append([r1, r2, r3])
            else:
                imagefile_dirc = create_coupled_traindata(TRAINING_DATASET_DIRC, IMAGES_DIRC, geometrical_info, resulted_crack_data, 1, scaler)
                logging.info(f"Image file and report created at: {imagefile_dirc}")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            continue