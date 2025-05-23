import os
#from scipy.interpolate import lagrange
from abaqus import mdb
from abaqusConstants import (
    AXISYMMETRIC, DEFORMABLE_BODY, DELETE, ON, DISPLACEMENT, LINEAR, MODE_INDEPENDENT,
    TOTAL, SOLVER_DEFAULT, UNIFORM, UNSET, COMPUTED, CAX4T, STANDARD, FINER,
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



def solve_polynomial_coeffs(R0, RD, D, C_val, curvature_x, S0, SD):
    """
    Solves for the coefficients a2, a3, a4 of the quartic polynomial:
    y(x) = a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
    given the constraints:
    y(0) = R0
    y'(0) = S0
    y(D) = RD
    y'(D) = SD
    y''(curvature_x) = C_val

    Returns:
        coeffs (list): [a0, a1, a2, a3, a4]
    """
    a0 = R0
    a1 = S0

    # Matrix A for [a2, a3, a4]:
    # Constraint 1: y(D) = RD
    # a2*D^2 + a3*D^3 + a4*D^4 = RD - a0 - a1*D
    # Constraint 2: y'(D) = SD
    # 2*a2*D + 3*a3*D^2 + 4*a4*D^3 = SD - a1
    # Constraint 3: y''(curvature_x) = C_val
    # 2*a2 + 6*a3*curvature_x + 12*a4*curvature_x^2 = C_val
    A = np.array([
        [D**2, D**3, D**4],
        [2*D,  3*D**2, 4*D**3],
        [2,    6*curvature_x, 12*curvature_x**2]
    ])

    # Vector b:
    b = np.array([
        RD - a0 - a1*D,  # From y(D) = RD
        SD - a1,         # From y'(D) = SD
        C_val            # From y''(curvature_x) = C_val
    ])

    try:
        # Solve A * [a2, a3, a4]^T = b
        a2_a3_a4 = np.linalg.solve(A, b)
        return [a0, a1, a2_a3_a4[0], a2_a3_a4[1], a2_a3_a4[2]]
    except np.linalg.LinAlgError:
        print(f"Singular matrix for C_val = {C_val} at curvature_x = {curvature_x}, S0={S0}, SD={SD}. Could not solve.")
        return None

def polynomial_radius(x, coeffs):
    """
    Evaluates the quartic polynomial y(x) given coefficients.
    coeffs = [a0, a1, a2, a3, a4]
    """
    a0, a1, a2, a3, a4 = coeffs
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

def fetch_info_from_json(working_dirc, filename):
    file_dirc = os.path.join(working_dirc, filename)
    with open(file_dirc, 'r') as f:
        info_dict = json.load(f)
    return info_dict

def run_simulation(working_dirc, geometrical_info, material_properties, thermal_cycle_info, mesh_info):
    # Change the working directory to WORKING_DIRC
    os.chdir(working_dirc)
    print(f"DEBUG: Changed working directory to {working_dirc}")

    # Unpack geometrical information
    R_SI = geometrical_info['R_SI']
    Y_SI = geometrical_info['Y_SI']
    R0 = geometrical_info['R0']
    RD = geometrical_info['RD']
    F1 = geometrical_info['F1']
    F2 = geometrical_info['F2']

    centerline_offset_top = geometrical_info['centerline_offset_top']
    centerline_offset_bottom = geometrical_info['centerline_offset_bottom']
    curvature_pos_factor = geometrical_info['curvature_pos_factor']
    C_val = geometrical_info['C_val']
    S0_val = geometrical_info['S0_val']
    SD_val = geometrical_info['SD_val']
    D = Y_SI//F1
    curvature_x = curvature_pos_factor * D

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
    
    x_plot = np.linspace(0, D, int(Y_SI)+1) # Depth points for plotting
    coeffs = solve_polynomial_coeffs(R0, RD,D, C_val, curvature_x, S0_val, SD_val)
    r_CU = polynomial_radius(x_plot, coeffs)
    r_CU = r_CU[::-1]
    #print(r_CU[-1],r_CU[0])

    ry_CU = [(F2*r_CU[int(y)], y) for y in np.linspace(0, Y_SI, 100, endpoint=True).tolist()]

    # Model creation
    mymodel = mdb.models['Model-1']

    cu_sketch = mymodel.ConstrainedSketch(name='revolve_profile', sheetSize=200.0)
    v1 = cu_sketch.Spline(ry_CU)
    v2 = cu_sketch.Line(point1=ry_CU[-1], point2=(0, Y_SI))
    v3 = cu_sketch.Line(point1=(0, Y_SI), point2=(0, 0))
    v4 = cu_sketch.Line(point1=(0, 0), point2=ry_CU[0])
    rotation_axis = cu_sketch.ConstructionLine(point1=(0, 0), point2=(0, Y_SI))
    cu_sketch.assignCenterline(rotation_axis)
    
    cu_part = mymodel.Part(name='Cu', dimensionality=AXISYMMETRIC, type=DEFORMABLE_BODY)
    cu_part.BaseShell(sketch=cu_sketch)
    
    si_sketch = mymodel.ConstrainedSketch(name='revolve_profile', sheetSize=200.0)
    v1 = si_sketch.Spline(ry_CU)
    v2 = si_sketch.Line(point1=ry_CU[-1], point2=(R_SI, Y_SI))
    v3 = si_sketch.Line(point1=(R_SI, Y_SI), point2=(R_SI, 0))
    v4 = si_sketch.Line(point1=(R_SI, 0), point2=ry_CU[0])
    rotation_axis = si_sketch.ConstructionLine(point1=(0, 0), point2=(0, Y_SI))
    si_sketch.assignCenterline(rotation_axis)

    si_part = mymodel.Part(name='Si', dimensionality=AXISYMMETRIC, type=DEFORMABLE_BODY)
    si_part.BaseShell(sketch=si_sketch)

    myassembly = mymodel.rootAssembly
    
    cu_instance = myassembly.Instance(name='Cu', part=cu_part, dependent=ON)
    si_instance = myassembly.Instance(name='Si', part=si_part, dependent=ON)

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
    # Calculate the midpoint of the ry_CU list
    mid_index = len(ry_CU) // 2
    mid_point = ry_CU[mid_index]    
    contactline_cu = cu_part.Set(name="ContactLineCu",
                                 edges=cu_part.edges.findAt(((mid_point[0], mid_point[1], 0),)))
    contactline_si = si_part.Set(name="ContactLineSi", 
                                 edges=si_part.edges.findAt(((mid_point[0], mid_point[1], 0),)))
    allsurface_cu = cu_part.Set(name="AllSurfaceCu", faces=cu_part.faces)
    allsurface_si = si_part.Set(name="AllSurfaceSi", faces=si_part.faces)
    lateralline_si = si_part.Set(name="LateralLineSi", 
                                 edges=si_part.edges.findAt(((R_SI, Y_SI/2, 0),)))
    centerline_cu = cu_part.Set(name="CenterLineCu",
                                edges=cu_part.edges.findAt(((0, Y_SI/2, 0),)))

    #
    #-------------------------------------------------
    # Add Tie Constraint
    mymodel.Tie(name='CuSiTie', 
        main=cu_instance.sets['ContactLineCu'], 
        secondary=si_instance.sets['ContactLineSi'], 
        positionToleranceMethod=COMPUTED, 
        adjust=ON, 
        tieRotations=ON)
    
    #-------------------------------------------------
    # Update Temperature BCs to reference sets in instances

    mymodel.TemperatureBC(name='CuTempBC', createStepName='TCTCondition', region=cu_instance.sets['AllSurfaceCu'], 
                          distributionType=UNIFORM, fieldName='', magnitude=1.0, amplitude='ThermalCycle')

    mymodel.TemperatureBC(name='SiTempBC', createStepName='TCTCondition', region=si_instance.sets['AllSurfaceSi'], 
                          distributionType=UNIFORM, fieldName='', magnitude=1.0, amplitude='ThermalCycle')

    # Add boundary condition to fix the lateral surfaces
    mymodel.DisplacementBC(name='FixedSiLateral', createStepName='Initial', region=si_instance.sets['LateralLineSi'], 
                           u1=0.0, u2=0.0, u3=0.0, 
                           ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                           amplitude=UNSET, fixed=ON, distributionType=UNIFORM, fieldName='')
    
    mymodel.DisplacementBC(name='FixedCuCenter', createStepName='Initial', region=cu_instance.sets['CenterLineCu'], 
                           u1=0.0, u2=UNSET, u3=0.0, 
                           ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                           amplitude=UNSET, fixed=ON, distributionType=UNIFORM, fieldName='')
    #-------------------------------------------------
    # Define Via section
    via_section = mymodel.HomogeneousSolidSection(name='Via', material='Cu', thickness=None)

    # Define Wafer section
    wafer_section = mymodel.HomogeneousSolidSection(name='wafer', material='Si', thickness=None)

    # Assign sections to parts
    cu_part.SectionAssignment(region=cu_part.sets['AllSurfaceCu'], sectionName='Via')
    si_part.SectionAssignment(region=si_part.sets['AllSurfaceSi'], sectionName='wafer')

    myassembly.regenerate() # 할 필요는 없지만 안전을 위해 assembly 최신화
    #-------------------------------------------------
    # Adjust mesh settings for CU and SI parts near the contact surface
    # Seed the edges near the contact surface with the same size

    # Define the element type for coupled temperature-displacement analysis
    cu_elem_type = ElemType(elemCode=CAX4T, elemLibrary=STANDARD)
    si_elem_type = ElemType(elemCode=CAX4T, elemLibrary=STANDARD)

    # Assign the element type to the regions
    cu_part.setElementType(regions=cu_part.sets['AllSurfaceCu'], elemTypes=(cu_elem_type,))
    si_part.setElementType(regions=si_part.sets['AllSurfaceSi'], elemTypes=(si_elem_type,))


    contact_edges_cu = cu_part.edges.findAt(((mid_point[0], mid_point[1], 0),))
    contact_edges_si = si_part.edges.findAt(((mid_point[0], mid_point[1], 0),))
    print(f"contact_edges_cu: {contact_edges_cu}")

    cu_part.seedEdgeBySize(edges=contact_edges_cu, size=mesh_info['cu_mesh_size'], deviationFactor=0.1, constraint=FINER)
    si_part.seedEdgeByBias(end1Edges=contact_edges_si, minSize=mesh_info['cu_mesh_size'], maxSize=mesh_info['si_mesh_size'], biasMethod=SINGLE)

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
geometrical_info = fetch_info_from_json(SIMU_DIRC, 'geometrical_info.json')
material_properties = fetch_info_from_json(SIMU_DIRC, 'material_properties.json')
thermal_cycle_info = fetch_info_from_json(SIMU_DIRC, 'thermal_cycle_info.json')
mesh_info = fetch_info_from_json(SIMU_DIRC, 'mesh_info.json')
run_simulation(SIMU_DIRC, geometrical_info, material_properties, thermal_cycle_info, mesh_info)
