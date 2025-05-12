import os
import numpy as np  # Ensure numpy is imported for P_TABLE_CU calculation

def initialize_directories():
    """
    Initialize directories for simulation and training dataset.
    """
    import os

    # Get the current working directory or fallback if __file__ is not defined
    try:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        CURRENT_DIR = os.getcwd()
    print(f"Current Directory: {CURRENT_DIR}")

    # Create SIMU_DIRC directory
    SIMU_DIRC = os.path.join(CURRENT_DIR, 'SIMU_DIRC')
    if not os.path.exists(SIMU_DIRC):
        os.makedirs(SIMU_DIRC)
        print(f"Created folder: {SIMU_DIRC}")
    else:
        for root, dirs, files in os.walk(SIMU_DIRC):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        print(f"Cleared all files in folder: {SIMU_DIRC}")

    # Create TRAINING_DATASET directory
    TRAINING_DATASET_DIR = os.path.join(CURRENT_DIR, 'TRAINING_DATASET')
    if not os.path.exists(TRAINING_DATASET_DIR):
        os.makedirs(TRAINING_DATASET_DIR)
        print(f"Created folder: {TRAINING_DATASET_DIR}")


    # Create IMAGES directory
    IMAGES_DIR = os.path.join(TRAINING_DATASET_DIR, 'IMAGES')
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
        print(f"Created folder: {IMAGES_DIR}")
    
    return CURRENT_DIR

def run_simulation(working_dirc, geometrical_info, material_properties, thermal_cycle_info):
    import os
    from scipy.interpolate import lagrange
    from abaqus import mdb
    from abaqusConstants import (
        THREE_D, DEFORMABLE_BODY, DELETE, ON, DISPLACEMENT, LINEAR, MODE_INDEPENDENT,
        TOTAL, SOLVER_DEFAULT, UNIFORM, UNSET, COMPUTED, C3D6T, STANDARD, FINER,
        ANALYSIS, PERCENTAGE, DOUBLE, SINGLE, OFF, DEFAULT
    )
    from mesh import ElemType
    import locale
    import step
    import interaction
    from regionToolset import Region

    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    # Change the working directory to WORKING_DIRC
    os.chdir(working_dirc)
    print(f"DEBUG: Changed working directory to {working_dirc}")

    # Unpack geometrical information
    R_SI = geometrical_info['R_SI']
    Y_SI = geometrical_info['Y_SI']
    R0_CU = geometrical_info['R0_CU']
    Y0_CU = geometrical_info['Y0_CU']

    # Unpack material properties
    E_CU = material_properties['E_CU']
    NU_CU = material_properties['NU_CU']
    CTE_CU = material_properties['CTE_CU']
    Y_CU = material_properties['Y_CU']
    P_TABLE_CU = material_properties['P_TABLE_CU']
    K_CU = material_properties['K_CU']
    C_CU = material_properties['C_CU']
    RHO_CU = material_properties['RHO_CU']
    DAMAGEINIT_TABLE_CU = material_properties['DAMAGEINIT_TABLE_CU']
    DAMAGEEVOL_TABLE_CU = material_properties['DAMAGEEVOL_TABLE_CU']
    E_SI = material_properties['E_SI']
    NU_SI = material_properties['NU_SI']
    CTE_SI = material_properties['CTE_SI']
    K_SI = material_properties['K_SI']
    C_SI = material_properties['C_SI']
    RHO_SI = material_properties['RHO_SI']

    # r-z relation
    r_CU = lagrange(Y0_CU, R0_CU)
    ry_CU = [(r_CU(y), y) for y in np.linspace(0, Y_SI, 100, endpoint=True).tolist()]

    # Model creation
    mymodel = mdb.models['Model-1']

    cu_sketch = mymodel.ConstrainedSketch(name='revolve_profile', sheetSize=200.0)
    v1 = cu_sketch.Spline(ry_CU)
    v2 = cu_sketch.Line(point1=ry_CU[-1], point2=(0, Y_SI))
    v3 = cu_sketch.Line(point1=(0, Y_SI), point2=(0, 0))
    v4 = cu_sketch.Line(point1=(0, 0), point2=ry_CU[0])
    rotation_axis = cu_sketch.ConstructionLine(point1=(0, 0), point2=(0, Y_SI))
    cu_sketch.assignCenterline(rotation_axis)

    cu_part = mymodel.Part(name='Cu', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    cu_part.BaseSolidRevolve(sketch=cu_sketch, angle=360.0)

    si_sketch = mymodel.ConstrainedSketch(name='Box_profile', sheetSize=200.0)
    si_sketch.rectangle(point1=(-R_SI, 0), point2=(R_SI, Y_SI))
    uncut_si_part = mymodel.Part(name='UncutSi', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    uncut_si_part.BaseSolidExtrude(sketch=si_sketch, depth=2 * R_SI)

    myassembly = mymodel.rootAssembly
    cu_instance = myassembly.Instance(name='Cu', part=cu_part, dependent=ON)
    uncut_si_instance = myassembly.Instance(name='UncutSi', part=uncut_si_part, dependent=ON)
    myassembly.translate(instanceList=('UncutSi',), vector=(0, 0, -R_SI))

    si_part = myassembly.PartFromBooleanCut(
        name='Si',
        instanceToBeCut=uncut_si_instance,
        cuttingInstances=(cu_instance,),
        originalInstances=DELETE
    )
    del mymodel.parts['UncutSi']

    cu_instance = myassembly.Instance(name='Cu', part=cu_part, dependent=ON)
    si_instance = myassembly.Instance(name='Si', part=si_part, dependent=ON)

    # Calculate the midpoint of the ry_CU list
    mid_index = len(ry_CU) // 2
    mid_point = ry_CU[mid_index]
    # 1. ContactSurf : Define contact surfaces for CU and SI parts
    contactsurf_cu = cu_part.Surface(name='ContactSurfCu', side1Faces=cu_part.faces.findAt(((mid_point[0], mid_point[1], 0),)))
    contactsurf_si = si_part.Surface(name='ContactSurfSi', side1Faces=si_part.faces.findAt(((mid_point[0], mid_point[1], 0),)))

    # 2. Volume : Create sets for all cells in Si and Cu parts
    si_part.Set(name='Sivolume', cells=si_part.cells[:])
    cu_part.Set(name='Cuvolume', cells=cu_part.cells[:])

    # Manually select the four lateral faces of Si
    left_face = si_part.faces.findAt(((-R_SI, Y_SI / 2, 0),))  # Left face
    right_face = si_part.faces.findAt(((R_SI, Y_SI / 2, 0),))  # Right face
    front_face = si_part.faces.findAt(((0, Y_SI / 2, -R_SI),))  # Front face
    back_face = si_part.faces.findAt(((0, Y_SI / 2, R_SI),))  # Back face
    # 3. SiLateralSurf : Combine the selected faces into a single set# Combine the selected faces into a single set
    si_part.Set(name='SiLateralSurf', faces=(left_face, right_face, front_face, back_face))


    #-------------------------------------------------
    # 구리 재료 정의
    cu_material = mymodel.Material(name = 'Cu')
    cu_material.Elastic(table=((E_CU, NU_CU),)) # Elastic Modulus and Poisson's Ratio
    cu_material.Plastic(table=P_TABLE_CU) # Plasticity true stress-strain curve
    cu_material.Expansion(table=((CTE_CU,),)) # Coefficient of Thermal Expansion
    cu_material.Density(table=((RHO_CU,),)) # Density
    cu_material.Conductivity(table=((K_CU,),)) # Thermal Conductivity
    cu_material.SpecificHeat(table=((C_CU,),)) # Specific Heat

    # Add ductile damage initiation to Cu material
    cu_material_fracturemode = cu_material.DuctileDamageInitiation(table=DAMAGEINIT_TABLE_CU)  # Example values: triaxiality, equivalent plastic strain, strain rate
    cu_material_fracturemode.DamageEvolution(type=DISPLACEMENT, softening=LINEAR, mixedModeBehavior=MODE_INDEPENDENT, table=DAMAGEEVOL_TABLE_CU)  # Example values: displacement, rate
    # 실리콘 재료 정의
    si_material = mymodel.Material(name='Si')
    si_material.Elastic(table=((E_SI, NU_SI),)) # Elastic Modulus and Poisson's Ratio
    si_material.Expansion(table=((CTE_SI,),)) # Coefficient of Thermal Expansion
    si_material.Density(table=((RHO_SI,),)) # Density
    si_material.Conductivity(table=((K_SI,),)) # Thermal Conductivity
    si_material.SpecificHeat(table=((C_SI,),)) # Specific Heat
    #-------------------------------------------------

    # Define Via section
    via_section = mymodel.HomogeneousSolidSection(name='Via', material='Cu', thickness=None)

    # Define Wafer section
    wafer_section = mymodel.HomogeneousSolidSection(name='wafer', material='Si', thickness=None)

    # Assign sections to parts
    cu_region = (cu_part.cells, )
    cu_part.SectionAssignment(region=cu_region, sectionName='Via')

    si_region = (si_part.cells, )
    si_part.SectionAssignment(region=si_region, sectionName='wafer')

    myassembly.regenerate() # 할 필요는 없지만 안전을 위해 assembly 최신화

    #-------------------------------------------------
    #Steps
    # 사용자 정의 Amplitude 생성 (온도 사이클)
    single_cycle_data = thermal_cycle_info['single_cycle_data']
    single_cycle_data_multi = single_cycle_data[1:]

    # 열사이클 반복 설정
    cycle_time = single_cycle_data[-1][0]  # 마지막 시간 값
    total_cycles = thermal_cycle_info['total_cycles']
    custom_amplitude_data = single_cycle_data
    for i in range(total_cycles-1):
        for time, temp in single_cycle_data_multi:
            custom_amplitude_data.append((time + (i+1) * cycle_time, temp))


    # Amplitude 정의
    mymodel.TabularAmplitude(name='ThermalCycle', timeSpan=TOTAL, smooth=SOLVER_DEFAULT, data=custom_amplitude_data)

    # Step 설정
    total_time = total_cycles * cycle_time

    mymodel.CoupledTempDisplacementStep(name='TCTCondition', previous='Initial', timePeriod=total_time, initialInc=0.1, minInc=1e-5, maxInc=1, deltmx=1, maxNumInc=100000)
    #-------------------------------------------------
    # Update Temperature BCs to reference sets in instances
    region_sivolume = myassembly.instances['Si'].sets['Sivolume']
    region_cuvolume = myassembly.instances['Cu'].sets['Cuvolume']

    mymodel.TemperatureBC(name='SiTempBC', createStepName='TCTCondition', region=region_sivolume, 
                          distributionType=UNIFORM, fieldName='', magnitude=1.0, amplitude='ThermalCycle')

    mymodel.TemperatureBC(name='CuTempBC', createStepName='TCTCondition', region=region_cuvolume, 
                          distributionType=UNIFORM, fieldName='', magnitude=1.0, amplitude='ThermalCycle')

    # Convert the set to a region
    region_silateralsurf = myassembly.instances['Si'].sets['SiLateralSurf']

    # Add boundary condition to fix the lateral surfaces
    mymodel.DisplacementBC(name='FixedSiLateral', createStepName='Initial', region=region_silateralsurf, 
                           u1=0.0, u2=0.0, u3=0.0, 
                           ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                           amplitude=UNSET, fixed=ON, distributionType=UNIFORM, fieldName='')



    # Update the CuSiTie constraint to use instance contact surfaces
    contactsurf_cu_instance = myassembly.instances['Cu'].surfaces['ContactSurfCu']
    contactsurf_si_instance = myassembly.instances['Si'].surfaces['ContactSurfSi']

    mymodel.Tie(name='CuSiTie', 
        main=contactsurf_cu_instance, 
        secondary=contactsurf_si_instance, 
        positionToleranceMethod=COMPUTED, 
        adjust=ON, 
        tieRotations=ON)



    # Adjust mesh settings for CU and SI parts near the contact surface
    # Seed the edges near the contact surface with the same size

    # Assign coupled temperature-displacement element type to CU and SI parts
    cu_region = (cu_part.cells, )
    si_region = (si_part.cells, )

    # Define the element type for coupled temperature-displacement analysis
    cu_elem_type = ElemType(elemCode=C3D6T, elemLibrary=STANDARD)
    si_elem_type = ElemType(elemCode=C3D6T, elemLibrary=STANDARD)

    # Assign the element type to the regions
    cu_part.setElementType(regions=cu_region, elemTypes=(cu_elem_type,))
    si_part.setElementType(regions=si_region, elemTypes=(si_elem_type,))


    contact_edges_cu = cu_part.edges.findAt(((mid_point[0], mid_point[1], 0),))
    contact_edges_si = si_part.edges.findAt(((mid_point[0], mid_point[1], 0),))

    cu_part.seedEdgeBySize(edges=contact_edges_cu, size=10.0, deviationFactor=0.1, constraint=FINER)
    si_part.seedEdgeBySize(edges=contact_edges_si, size=10.0, deviationFactor=0.1, constraint=FINER)

    # Generate the meshes
    cu_part.generateMesh()
    si_part.generateMesh()

    print(f"DEBUG: WORKING_DIRC set to {working_dirc}")
    # Create a job for the analysis
    mdb.Job(name='ThermalAnalysis', model='Model-1', description='Thermal and mechanical analysis', 
             type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
             memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=DOUBLE, 
             nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, 
             historyPrint=OFF, userSubroutine='', scratch=working_dirc, multiprocessingMode=DEFAULT, numCpus=1, numDomains=1, numGPUs=0)

    mdb.saveAs(pathName=os.path.join(working_dirc, 'auto_model_output.cae'))

    # Submit the job and wait for completion
    mdb.jobs['ThermalAnalysis'].submit(consistencyChecking=OFF)
    mdb.jobs['ThermalAnalysis'].waitForCompletion()

    # Check if the ODB file was successfully created
    odb_path = os.path.join(working_dirc, 'ThermalAnalysis.odb')
    if os.path.exists(odb_path):
        return odb_path
    else:
        return None  # Signal that something went wrong
    
def save_initial_coordinates(odb_path, output_file):
    from odbAccess import openOdb
    import os

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
    from odbAccess import openOdb
    import os

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
                with open(csv_path, 'w') as csv_file:
                    csv_file.write(f"{image_dirc}, {crack_time}, {r_coord}, {y_coord}\n")
                



CURRENT_DIR = initialize_directories()
'''
디버그용 current directory 읽는 코드
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
'''
SIMU_DIRC = os.path.join(CURRENT_DIR, 'SIMU_DIRC')
TRAINING_DATASET_DIR = os.path.join(CURRENT_DIR, 'TRAINING_DATASET')
IMAGES_DIRC = os.path.join(TRAINING_DATASET_DIR, 'IMAGES')


geometry_info = {
    'R_SI': 100,
    'Y_SI': 100,
    'R0_CU': [15, 15, 10],
    'Y0_CU': [0, 60, 100]
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
        (18.33, 300.0),    # 15도/분으로 가열하여 18.33분 후 300°C 도달
        (28.33, 300.0),    # 10분간 300°C 유지
        (44.33, -65.0),    # -10도/분으로 냉각하여 44.33분 후 -65°C 도달
        (54.33, -65.0),    # 10분간 -65°C 유지
        (70.0, 25.0)       # 15도/분으로 가열하여 70분 후 25°C 도달
    ],
    'total_cycles': 10
}
output_odb_dirc = run_simulation(SIMU_DIRC, geometry_info, material_properties, thermal_cycle_info)
#디버그용 odb 파일 경로 읽는 코드
#output_odb_dirc = os.path.join(SIMU_DIRC, 'ThermalAnalysis.odb')

save_initial_coordinates(output_odb_dirc, os.path.join(SIMU_DIRC, 'initial_coordinates.txt'))
extract_failed_elements(output_odb_dirc, os.path.join(SIMU_DIRC, 'failed_elements.txt'))
scaler = 10
image_dirc = create_part_images(IMAGES_DIRC, geometry_info, scaler)
couple_crackresult_to_images(TRAINING_DATASET_DIR, image_dirc, SIMU_DIRC, scaler)