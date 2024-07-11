"""storefront URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
#importing the functions from other modules urls will take the user to
from django.contrib import admin
from django.urls import path, include
from pages.views import home_view, about_view, optimization_view, operations_view, plot_view, calculations_view, results_view, demographic_view, your_form_submission_view

#set up for where each url takes the user
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name = 'home'),
    path('about/', about_view, name = 'about'),
    path('optimization_tool/', optimization_view, name = 'optimization_tool'), #asks questions to user
    path('operations/', operations_view, name = 'operations'),
    path('optimization_tool/results_page', results_view, name='results'), # shows results
    path('calculator/plot', plot_view, name = 'plot'), #shows graphs
    path('optimization_tool/optimization', calculations_view, name = 'optimization'), #shows results
    path('demographic/', demographic_view, name='demographic'), #shows demographic data
    path('form_submission/', your_form_submission_view, name='your_form_submission_view'),
   

]
#to run Django: python manage.py runserver

