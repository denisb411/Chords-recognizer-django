# -*- coding: utf-8 -*-
# Generated by Django 1.11.4 on 2017-09-25 02:51
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Predicted',
        ),
    ]