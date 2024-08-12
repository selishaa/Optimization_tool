# test.py
from django.test import RequestFactory
from GeneticAlgorithmPython.Finaloneyear_ST import main as mainYear

# Create a request object using RequestFactory
request_factory = RequestFactory()
request = request_factory.get('/some-url/')

# Pass the mock request object to the mainYear function
mainYear(request)
