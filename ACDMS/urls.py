"""ACDMS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from ACDMSApp import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path('admin/', admin.site.urls),
    path('getStarted/', views.getStarted, name='getStarted'),
    path('', views.getStarted, name='getStarted'),
    path('home/', views.Home, name='Home'),
    path('addNewRecord/', views.addNewRecord, name='addNewRecord'),
    path('login/', views.login, name='login'),
    path('saveRecord/', views.saveRecord, name='saveRecord'),
    path('automaticPatientInfoFill/', views.automaticPatientInfoFill,
         name='automaticPatientInfoFill'),
    path('processingPapSmearImage/', views.processingPapSmearImage,
         name='processingPapSmearImage'),
    path('processImageBreastCancerBinary/', views.processImageBreastCancerBinary,
         name='processImageBreastCancerBinary'),
    path('savePetientInfo/', views.savePetientInfo, name='savePetientInfo'),
    path('getPatientInfor/', views.getPatientInfor, name='getPatientInfor'),
]+static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
