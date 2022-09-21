from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(pathologist_account, pathologist_account_admin)
admin.site.register(patient_info)
admin.site.register(patient_testDetails)