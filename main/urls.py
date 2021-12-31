from django.urls import path, include
from .views import *

# app_name = "main"

urlpatterns = [
    path('', index, name='gg'),
    path('Home', Home, name='Home')
]