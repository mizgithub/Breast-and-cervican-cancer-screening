from django.db import models
from django.contrib import admin
# Create your models here.
class pathologist_account(models.Model):
	account_stat = (('Enabled',('Enabled')), ('Disabled',('Disabled')))
	fullName = models.CharField(max_length=200)
	username = models.CharField(max_length=200)
	password = models.CharField(max_length=200)
	date_created = models.DateField(auto_now_add=True)
	account_status = models.CharField(max_length=10, choices = account_stat, default='Enabled')
	def __str__(self):
		return self.fullName
class pathologist_account_admin(admin.ModelAdmin):
	list_display = ('fullName', 'username', 'account_status')
	search_fields =['fullName', 'username']
class patient_info(models.Model):
	patient_id = models.CharField(max_length=200)
	patient_fullName = models.CharField(max_length=200)
	patient_address = models.CharField(max_length = 200)
	patient_phone = models.CharField(max_length = 13)
	patient_dateOfBirth = models.DateField()
class patient_testDetails(models.Model):
	patient = models.ForeignKey(patient_info, on_delete = models.CASCADE)
	recording_pathologist_name = models.CharField(max_length = 200)
	sample_type = models.CharField(max_length = 200)
	cancer_name = models.CharField(max_length = 20)
	cancer_type = models.CharField(max_length = 20)
	cancer_subType = models.CharField(max_length = 20)
	cancer_grade = models.CharField(max_length = 20)
	pathologist_remark = models.TextField()
	record_date = models.DateField()
	record_time = models.TimeField()
	sample_imageFile = models.TextField()
