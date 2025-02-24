from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.index, name='predict_image'),  # Using the same view for form submission
]
