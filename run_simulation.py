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
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import step
import interaction
from regionToolset import Region
import json
import numpy as np

def fetch_info_from_json(working_dirc, filename):
    file_dirc = os.path.join(working_dirc, filename)
    with open(file_dirc, 'r') as f:
        info_dict = json.load(f)
    return info_dict

def run_simulation(working_dirc, geometry_info, material_properties, thermal_cycle_info):
    # Change the working directory to WORKING_DIRC
    os.chdir(working_dirc)
    print(f"DEBUG: Changed working directory to {working_dirc}")

    # Unpack geometrical information
    R_SI = geometry_info['R_SI']
    Y_SI = geometry_info['Y_SI']
    R0_CU = geometry_info['R0_CU']
    Y0_CU = geometry_info['Y0_CU']

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
 
# Set CURRENT_DIRC to the directory of this script using the direct file name
CURRENT_DIRC = os.path.dirname(os.path.abspath('run_simulation.py'))

path_dict = fetch_info_from_json(CURRENT_DIRC, 'path_info.json')
SIMU_DIRC = path_dict['SIMU_DIRC']
geometry_info = fetch_info_from_json(SIMU_DIRC, 'geometry_info.json')
material_properties = fetch_info_from_json(SIMU_DIRC, 'material_properties.json')
thermal_cycle_info = fetch_info_from_json(SIMU_DIRC, 'thermal_cycle_info.json')
run_simulation(SIMU_DIRC, geometry_info, material_properties, thermal_cycle_info)