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
from GeneticAlgorithmPython.Finaloneyear_ST import bestYear, LPSP, LWSP, return_graph1, return_graph2, return_graph3, return_graph4, return_graph5, return_graph6, return_graph7 





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
    return HttpResponse("<h1>Renewable Energy</h1>") #string of HTML code

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

def your_form_submission_view(request):
    if request.method == 'POST':
        # Process the form data here
        population = request.POST.get('population')
        household = request.POST.get('household')
        acre = request.POST.get('acre')
        cattle = request.POST.get('cattle')
        
       
        
        return redirect('home')  # Redirect to home or any other page after processing
    else:
        return redirect('demographic')



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
    mainYear()
    #solar, wind, hydropower, biogas, battery, biogas-powered water pump, wind-powered water pump, solar-powered water pump
    individual = bestYear()
    
    ACS = 26865.804379422112  # Replace with your actual ACS calculation
    Initial_cost = 15814.384299089257  # Replace with your actual Initial cost calculation
    NPC = 511408.13195292  # Replace with your actual NPC calculation
    CO2_emitted = 1.600643819103928  # Replace with your actual CO2 emitted calculation
    HDI = 0.3806041481619234  # Replace with your actual HDI calculation
    Eexcess = -312393.5669519148

  
    LPSP_print = LPSP


    chart1 = return_graph1
    chart2 = return_graph2
    chart3 = return_graph3
    chart4 = return_graph4
    chart5 = return_graph5
    chart6 = return_graph6
    chart7 = return_graph7
    
    
    
    
  
   

    
    #sends the values to the results page
    return render(request, 'result_page.html', {
    "ACS": ACS,
    "Initial_cost": Initial_cost,
    "NPC": NPC,
    "CO2_emitted": CO2_emitted,
     "HDI": HDI,
     "Eexcess": Eexcess,
     "individual": individual,
     "LPSP_print": LPSP_print,
    "chart1": chart1, "chart2": chart2, "chart3": chart3, "chart4": chart4, "chart5": chart5, "chart6": chart6, "chart7": chart7
   
    })
   


    # def results_view(request):
    #     results = request.session.get('results', {})
        
    #     # Assuming chart1, chart2, ... are base64 encoded strings in results
    #     chart1 = results.get('chart1', None)
    #     chart2 = results.get('chart2', None)
    #     chart3 = results.get('chart3', None)
    #     chart4 = results.get('chart4', None)
    #     chart5 = results.get('chart5', None)

    #     # Decode charts if they are present
    #     # Decode the base64 strings if they exist
    #     if chart1:
    #         try:
    #             chart1_decoded = base64.b64decode(chart1).decode('utf-8')
    #         except Exception as e:
    #             print(f"Error decoding chart1: {e}")
    #             chart1_decoded = None
    #     else:
    #         chart1_decoded = None

    #     if chart2:
    #         try:
    #             chart2_decoded = base64.b64decode(chart2).decode('utf-8')
    #         except Exception as e:
    #             print(f"Error decoding chart2: {e}")
    #             chart2_decoded = None
    #     else:
    #         chart2_decoded = None

    #     if chart3:
    #         try:
    #             chart3_decoded = base64.b64decode(chart3).decode('utf-8')
    #         except Exception as e:
    #             print(f"Error decoding chart3: {e}")
    #             chart3_decoded = None
    #     else:
    #         chart3_decoded = None

    #     if chart4:
    #         try:
    #             chart4_decoded = base64.b64decode(chart4).decode('utf-8')
    #         except Exception as e:
    #             print(f"Error decoding chart4: {e}")
    #             chart4_decoded = None
    #     else:
    #         chart4_decoded = None

    #     if chart5:
    #         try:
    #             chart5_decoded = base64.b64decode(chart5).decode('utf-8')
    #         except Exception as e:
    #             print(f"Error decoding chart5: {e}")
    #             chart5_decoded = None
    #     else:
    #         chart5_decoded = None

    #     return render(request, 'result_page.html', {
    #         'solar_energy': results.get('solar_energy', None),
    #         'solar': results.get('solar', None),
    #         'wind_energy': results.get('wind_energy', None),
    #         'wind': results.get('wind', None),
    #         'hydro_energy': results.get('hydro_energy', None),
    #         'hydro': results.get('hydro', None),
    #         'bio_energy': results.get('bio_energy', None),
    #         'bio': results.get('bio', None),
    #         'solar_ind': results.get('solar_ind', None),
    #         'wind_ind': results.get('wind_ind', None),
    #         'hydropower_ind': results.get('hydropower_ind', None),
    #         'biogas_ind': results.get('biogas_ind', None),
    #         'battery_ind': results.get('battery_ind', None),
    #         'biogasWP_ind': results.get('biogasWP_ind', None),
    #         'windWP_ind': results.get('windWP_ind', None),
    #         'solarWP_ind': results.get('solarWP_ind', None),
    #         'LPSP': results.get('LPSP', None),
    #         'LWSP': results.get('LWSP', None),
    #         'chart1': chart1_decoded,
    #         'chart2': chart2_decoded,
    #         'chart3': chart3_decoded,
    #         'chart4': chart4_decoded,
    #         'chart5': chart5_decoded,
    #     })
        
    
# def calculations_view(request):
#     if request.method == 'POST':
#         print("Post request at calculations_view is working")

 
#         solar = str(request.POST.get("solar"))
#         if solar == "sunny":
#             G = [0, 0, 0, 0, 0, 1, 93, 273, 470, 664, 819, 932, 975, 992, 762, 820, 598, 459, 261, 25, 0, 0, 0, 0]
#         elif solar == "partly_sunny":
#             G = [0, 0, 0, 0, 0, 0, 0, 9, 310, 95, 353, 646, 692, 675, 590, 458, 300, 77, 1, 0, 0, 0, 0, 0]
#         elif solar == "cloudy":
#             G = [0, 0, 0, 0, 0, 0, 0, 13, 144, 331, 527, 425, 361, 392, 288, 212, 144, 79, 24, 0, 0, 0, 0, 0]
#         else:
#             G = []
#         set_solar(G)
#         solar_energy = get_solar()

#         wind = str(request.POST.get("wind"))
#         if wind == "high_wind":
#             v = [7, 6, 9, 9, 11, 10, 10, 11, 18, 15, 14, 15, 11, 16, 15, 14, 15, 13, 16, 14, 16, 15, 13, 12]
#         elif wind == "medium_wind":
#             v = [10, 9, 8, 7, 6, 7, 5, 9, 7, 10, 9, 10, 9, 7, 6, 6, 8, 7, 5, 4, 1, 1, 1, 2]
#         elif wind == "low_wind":
#             v = [5, 2, 1, 1, 0, 1, 1, 2, 3, 6, 6, 4, 4, 3, 5, 5, 1, 6, 2, 5, 2, 5, 4, 4]
#         else:
#             v = []
#         set_wind(v)
#         wind_energy = get_wind()

#         lakes = 'lakes' in request.POST
#         rivers = 'rivers' in request.POST
#         ponds = 'ponds' in request.POST

#         if rivers and lakes and ponds:
#             base_h = 35
#             hydro = "lakes, rivers and ponds"
#         elif rivers and lakes and not ponds:
#             base_h = 25
#             hydro = "lakes and rivers"
#         elif not rivers and lakes and ponds:
#             base_h = 10
#             hydro = "lakes and ponds"
#         elif rivers and not lakes and not ponds:
#             base_h = 20
#             hydro = "rivers"
#         elif lakes and not rivers and not ponds:
#             base_h = 5
#             hydro = "lakes"
#         elif ponds and not rivers and not lakes:
#             base_h = 2
#             hydro = "ponds"
#         else:
#             base_h = 0
#             hydro = ""
#         set_hydro(base_h)
#         hydro_energy = get_hydro()

#         cattle = 'cattle' in request.POST
#         agriculture = 'agriculture' in request.POST

#         if cattle and agriculture:
#             base_b = 2
#             bio = "cattle and agriculture"
#         elif cattle or agriculture:
#             base_b = 1
#             bio = "cattle or agriculture"
#         else:
#             base_b = 0
#             bio = ""
#         set_bio(base_b)
#         bio_energy = get_bio()

#         mainYear()
#         individual = bestYear()
#         solar_ind = individual[0]
#         wind_ind = individual[1]
#         hydropower_ind = individual[2]
#         biogas_ind = individual[3]
#         battery_ind = individual[4]
#         biogasWP_ind = individual[5]
#         windWP_ind = individual[6]
#         solarWP_ind = individual[7]

#         LWSP_print = LWSP()
#         LPSP_print = LPSP()

#         # Convert chart images to base64 strings (dummy data for now)
#         chart1 = b64encode(b'sample_chart_data1').decode('utf-8')
#         chart2 = b64encode(b'sample_chart_data2').decode('utf-8')
#         chart3 = b64encode(b'sample_chart_data3').decode('utf-8')
#         chart4 = b64encode(b'sample_chart_data4').decode('utf-8')
#         chart5 = b64encode(b'sample_chart_data5').decode('utf-8')

#         # Store the calculated data in the session
#         results = {
#             "solar_energy": solar_energy, "solar": solar,
#             "wind": wind, "wind_energy": wind_energy,
#             "hydro": hydro, "hydro_energy": hydro_energy,
#             "bio": bio, "bio_energy": bio_energy,
#             "LPSP": LPSP_print, "LWSP": LWSP_print,
#             "individual": individual,
#             "solar_ind": solar_ind, "wind_ind": wind_ind, 'hydropower_ind': hydropower_ind,
#             "biogas_ind": biogas_ind, "battery_ind": battery_ind,
#             "biogasWP_ind": biogasWP_ind, "windWP_ind": windWP_ind, "solarWP_ind": solarWP_ind,
#             "chart1": chart1, "chart2": chart2, "chart3": chart3, "chart4": chart4, "chart5": chart5
#         }
        
        
        

       

#         request.session['results'] = results

#         # Send JSON response indicating the calculation is done
#         return JsonResponse({"status": "ok", "url": "results_page"})

#     return render(request, 'optimization_tool.html')


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
    mainYear()
    individual = bestYear()
    solar_ind, wind_ind, hydropower_ind, biogas_ind, battery_ind, biogasWP_ind, windWP_ind, solarWP_ind = individual

    # Get charts from the Genetic Algorithm module
    chart1 = return_graph1()
    chart2 = return_graph2()
    chart3 = return_graph3()
    chart4 = return_graph4()
    chart5 = return_graph5()

   
