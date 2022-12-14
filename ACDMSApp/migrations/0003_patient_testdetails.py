# Generated by Django 3.1.7 on 2021-03-23 16:43

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ACDMSApp', '0002_patient_info'),
    ]

    operations = [
        migrations.CreateModel(
            name='patient_testDetails',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('recording_pathologist_name', models.CharField(max_length=200)),
                ('sample_type', models.CharField(max_length=200)),
                ('cancer_name', models.CharField(max_length=20)),
                ('cancer_type', models.CharField(max_length=20)),
                ('cancer_subType', models.CharField(max_length=20)),
                ('cancer_grade', models.CharField(max_length=20)),
                ('pathologist_remark', models.TextField()),
                ('record_date', models.DateField()),
                ('record_time', models.TimeField()),
                ('sample_imageFile', models.TextField()),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ACDMSApp.patient_info')),
            ],
        ),
    ]
