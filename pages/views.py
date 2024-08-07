#dir to begin server is /storefront
#importing functions from other modules that will be called in this one 
from django.shortcuts import render, redirect
from django.http import HttpResponse
import matplotlib
matplotlib.use('Agg') 
from .utils import get_plot
import matplotlib.pyplot as plt
import numpy as np
from .SIRES import set_solar, get_solar, set_hydro, get_hydro, set_bio, get_bio, set_wind, get_wind 
#from .SIRES_Final import main
from .NoElitismTrial import main, best 

# GeneticAlgorithmPython.Finaloneyear_ST
from GeneticAlgorithmPython.Finaloneyear_ST import main as mainYear
from GeneticAlgorithmPython.Finaloneyear_ST import bestYear, LPSP, LWSP,  return_graph2, return_graph3, return_graph4, return_graph5, return_graph6, return_graph7 

from GeneticAlgorithmPython.Microgrid_with_diesel_ST import fig1, fig2, fig3
from GeneticAlgorithmPython.Microgrid_without_diesel_ST import figwd1, figwd2, figwd3







from base64 import b64encode
from django.http import JsonResponse

import json
from base64 import b64encode
import os
import numpy as np
import base64
# Create your views here.



def home_view(request, *args, **kwargs):
    print(request.user)
    return render(request, 'home.html', )

def contact_view(request, *args, **kwargs):
    return render(request, 'contact.html')

def about_view(request):
    my_context = {
        "my_team" : "This is about the people working on the project",
        "my_number" : 2023,
        "my_list" : [1,2,3],
        "my_values" : ["a", "b", "c"]
    }
    return render(request, 'about.html', my_context)

def optimization_view(request):
    return render(request, 'optimization.html', {})

def demographic_view(request):
    return render(request, 'demo.html', {})

session_data = {
    'population': None,
    'household': None,
    'hectors': None,
    'cattle': None
}

from django.shortcuts import render

def comparison(request):
    # Optimized values
    ind = [1, 2, 3, 4, 5, 6, 7, 8]
    mainYear(request)
    # Solar, wind, hydropower, biogas, battery, biogas-powered water pump, wind-powered water pump, solar-powered water pump
    individual = bestYear()
    
    # SIRES data
    ACS = request.session.get('ACS')
    Initial_cost = request.session.get('InitialCost')
    NPC = request.session.get('NPC')
    CO2_emitted = request.session.get('CO2_total')
    HDI = request.session.get('HDI_total')
    
    
    # Microgrid with diesel data
    ACS1 = 27334.18616181368
    Initial_cost1 = 15259.809292713882
    NPC1 = 520324.0850727574
    CO2_emitted1 = 3.392324490149929
    HDI1 = 0.33732247130951326
    
    
    # Microgrid without diesel data
    ACSwd = 26489.274086932433
    Initial_costwd = 15407.817652552336
    NPCwd = 504240.63192997914
    CO2_emittedwd = 1.8336755819857073
    HDIwd = 0.3742939469592647
    
    
    
    print("ACS: ", ACS)
    print("Initial_cost: ", Initial_cost)
    print("NPC: ", NPC)
    print("CO2_emitted: ", CO2_emitted)
    print("HDI: ", HDI)
    
    chart3 = return_graph3()  # Ensure these functions are defined
    chart5 = return_graph5()
    chart6 = return_graph6()

    fig1_data = fig1()
    fig2_data = fig2()
    fig3_data = fig3()
    figwd1_data = figwd1()  
    figwd2_data = figwd2()
    figwd3_data = figwd3()
    
    return render(request, 'comparison.html', {
        "fig1": fig1_data,
        "fig2": fig2_data,
        "fig3": fig3_data,
        "figwd1": figwd1_data,  
        "figwd2": figwd2_data,      
        "figwd3": figwd3_data,
        "ACS": ACS,
        "Initial_cost": Initial_cost,
        "NPC": NPC,
        "CO2_emitted": CO2_emitted,
        "HDI": HDI,
        
        "chart3": chart3,   
        "chart5": chart5,
        "chart6": chart6,
        "ACS1": ACS1,     
        "Initial_cost1": Initial_cost1,
        "NPC1": NPC1,
        "CO2_emitted1": CO2_emitted1,
        "HDI1": HDI1,
        
        "ACSwd": ACSwd,
        "Initial_costwd": Initial_costwd,
        "NPCwd": NPCwd,
        "CO2_emittedwd": CO2_emittedwd,
        "HDIwd": HDIwd,
        
    })




def your_form_submission_view(request):
    if request.method == 'POST':
        # Process the form data here
        request.session['population'] = request.POST.get('population')
        request.session['household'] = request.POST.get('household')
        request.session['hectors'] = request.POST.get('hectors')
        request.session['cattle'] = request.POST.get('cattle')
        
        return redirect('optimization_tool')  # Redirect to the next view
    return render(request, 'demo.html', {
        
    })




#trial of a graph
def plot_view(request):
    #x = [1, 2, 3]
    #y = [2, 4, 6]
    x = np.arange(0, 2*(np.pi), 0.1)
    y = np.cos(x)
    chart = get_plot(x, y)
    return render(request, 'result.html', {'chart': chart})

def results_view(request):
     results = request.session.get('results', {})
     return render(request, 'result_page.html', results)
 
 


#calls the functions in SIRES with the given values input by the user
def calculations_view(request):
    #Solar
    
    
    solar = str(request.GET["solar"])
    if(solar == "sunny"):
        G = [0,0,0,0,0,1,93,273,470,664,819,932,975,992,762,820,598,459,261,25, 0,0,0,0]    
    elif(solar == "partly_sunny"):
        G = [0,0,0,0,0,0,0,9,310,95,353,646,692,675,590,458,300,77,1,0,0,0,0,0]
    elif(solar == "cloudy"):
        G = [0,0,0,0,0,0,0,13,144,331,527,425,361,392,288,212,144,79,24,0,0,0,0,0]
    set_solar(G)
    solar_energy = get_solar
    solar_energy_value = get_solar()

    #Wind
    wind = str(request.GET["wind"])
    if(wind == "high_wind"):
        v= [7,6,9,9,11,10,10,11,18,15,14,15,11,16,15,14,15,13,16,14,16,15,13,12]    
    elif(wind == "medium_wind"):
        v= [10,9,8,7,6,7,5,9,7,10,9,10,9,7,6,6,8,7,5,4,1,1,1,2]
    elif(wind == "low_wind"):
        v= [5,2,1,1,0,1,1,2,3,6,6,4,4,3,5,5,1,6,2,5,2,5,4,4]
    set_wind(v)
    wind_energy = get_wind
    wind_energy_value = get_wind()
    
    #Hydro
    if('lakes' in request.GET):
        lakes = request.GET['lakes']
    else:
        lakes = False
    if('rivers' in request.GET):
        rivers = request.GET['rivers']
    else:
        rivers = False
    if('ponds' in request.GET):
        ponds = request.GET['ponds']
    else:
        ponds = False

    if(rivers and lakes and ponds):
        base_h = 35
        hydro = "lakes, rivers and ponds"
    elif(rivers and lakes and not(ponds)):
        base_h = 25
        hydro = "lakes and rivers"
    elif(not(rivers) and lakes and ponds):
        base_h = 10
        hydro = "lakes and ponds"
    elif(rivers and not(lakes) and not(ponds)):
        base_h = 20
        hydro = "rivers"
    elif(lakes and not(rivers) and not(ponds)):
        base_h = 5
        hydro = "lakes"
    elif(ponds and not(rivers) and not(lakes)):
        base_h = 2
        hydro = "ponds"
    set_hydro(base_h)
    hydro_energy = get_hydro
    hydro_energy_value = get_hydro()
    
    #Biogas
    if('cattle' in request.GET):
        cattle = request.GET['cattle']
    else:
        cattle = False
    if('agriculture' in request.GET):
        agriculture = request.GET['agriculture']
    else:
        agriculture = False
    if(cattle and agriculture):
        base_b = 2
        bio = "cattle and agriculture"
    if((cattle and not(agriculture)) or (not(cattle) and agriculture)):
        base_b = 1
        bio = "cattle or agriculture"
    set_bio(base_b)
    bio_energy = get_bio
    bio_energy_value = get_bio()

    # #Optimized values

    ind = [1, 2, 3, 4, 5, 6, 7, 8]
    mainYear(request)
    #solar, wind, hydropower, biogas, battery, biogas-powered water pump, wind-powered water pump, solar-powered water pump
    individual = bestYear()
    
    
    
    ACS = request.session.get('ACS')
    Initial_cost = request.session.get('InitialCost')
    NPC = request.session.get('NPC')
    CO2_emitted = request.session.get('CO2_total')
    HDI = request.session.get('HDI_total')
    Eexcess = request.session.get('Eexcess')
    
    print("Eexcess: ", Eexcess)
    print("ACS: ", ACS)
    print("Initial_cost: ", Initial_cost)
    print("NPC: ", NPC)
    print("CO2_emitted: ", CO2_emitted)
    print("HDI: ", HDI)

  
    LPSP_print = LPSP


   
    chart2 = return_graph2
    chart3 = return_graph3
    chart4 = return_graph4
    chart5 = return_graph5
    chart6 = return_graph6
    chart7 = return_graph7
    
    
    
    
  
   

    
    #sends the values to the results page
    return render(request, 'result_page.html', {
    "solar_energy": solar_energy, "solar": solar, 
    "wind": wind, "wind_energy": wind_energy,
    "hydro": hydro, "hydro_energy": hydro_energy,
    "bio": bio, "bio_energy": bio_energy,
        
    "ACS": ACS,
    "Initial_cost": Initial_cost,
    "NPC": NPC,
    "CO2_emitted": CO2_emitted,
     "HDI": HDI,
     "Eexcess": Eexcess,
     "individual": individual,
     
     
     "LPSP_print": LPSP_print,
     "chart2": chart2, "chart3": chart3, "chart4": chart4, "chart5": chart5, "chart6": chart6, "chart7": chart7
   
    })
   




def operations_view(request):
    num_one = 15
    num_two = 12
    addition = num_one + num_two
    my_nums = {
        "num_one" : num_one,
        "num_two" : num_two,
        "addition" : addition
    }
    return render(request, 'operations.html', my_nums)




def calculate_renewable_energy(request):
    # Solar
    solar = str(request.GET.get("solar", ""))
    if solar == "sunny":
        G = [0, 0, 0, 0, 0, 1, 93, 273, 470, 664, 819, 932, 975, 992, 762, 820, 598, 459, 261, 25, 0, 0, 0, 0]
    elif solar == "partly_sunny":
        G = [0, 0, 0, 0, 0, 0, 0, 9, 310, 95, 353, 646, 692, 675, 590, 458, 300, 77, 1, 0, 0, 0, 0, 0]
    elif solar == "cloudy":
        G = [0, 0, 0, 0, 0, 0, 0, 13, 144, 331, 527, 425, 361, 392, 288, 212, 144, 79, 24, 0, 0, 0, 0, 0]
    set_solar(G)
    solar_energy = get_solar()

    # Wind
    wind = str(request.GET.get("wind", ""))
    if wind == "high_wind":
        v = [7, 6, 9, 9, 11, 10, 10, 11, 18, 15, 14, 15, 11, 16, 15, 14, 15, 13, 16, 14, 16, 15, 13, 12]
    elif wind == "medium_wind":
        v = [10, 9, 8, 7, 6, 7, 5, 9, 7, 10, 9, 10, 9, 7, 6, 6, 8, 7, 5, 4, 1, 1, 1, 2]
    elif wind == "low_wind":
        v = [5, 2, 1, 1, 0, 1, 1, 2, 3, 6, 6, 4, 4, 3, 5, 5, 1, 6, 2, 5, 2, 5, 4, 4]
    set_wind(v)
    wind_energy = get_wind()

    # Hydro
    lakes = 'lakes' in request.GET
    rivers = 'rivers' in request.GET
    ponds = 'ponds' in request.GET

    if rivers and lakes and ponds:
        base_h = 35
        hydro = "lakes, rivers and ponds"
    elif rivers and lakes and not ponds:
        base_h = 25
        hydro = "lakes and rivers"
    elif not rivers and lakes and ponds:
        base_h = 10
        hydro = "lakes and ponds"
    elif rivers and not lakes and not ponds:
        base_h = 20
        hydro = "rivers"
    elif lakes and not rivers and not ponds:
        base_h = 5
        hydro = "lakes"
    elif ponds and not rivers and not lakes:
        base_h = 2
        hydro = "ponds"
    set_hydro(base_h)
    hydro_energy = get_hydro()

    # Biogas
    cattle = 'cattle' in request.GET
    agriculture = 'agriculture' in request.GET

    if cattle and agriculture:
        base_b = 2
        bio = "cattle and agriculture"
    elif cattle or agriculture:
        base_b = 1
        bio = "cattle or agriculture"
    else:
        base_b = 0
        bio = "none"
    set_bio(base_b)
    bio_energy = get_bio()

    # Optimized values
    mainYear(request)
    individual = bestYear()
    solar_ind, wind_ind, hydropower_ind, biogas_ind, battery_ind, biogasWP_ind, windWP_ind, solarWP_ind = individual

    # Get charts from the Genetic Algorithm module
   
    chart2 = return_graph2()
    chart3 = return_graph3()
    chart4 = return_graph4()
    chart5 = return_graph5()

   
