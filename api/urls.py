#from django.conf.urls import url, patterns, include
from django.contrib import admin
from django.conf.urls import *

from . import views

from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    url(r'^clear/cached_data/$', views.clear_cached_data, name='clear_cached_data'),
    url(r'^clear/main_data/$', views.clear_main_data, name='clear_main_data'),
    url(r'^append/main_data/$', views.append_to_main_data, name='append_to_main_data'),    
    url(r'^append/cached_data/$', views.append_to_cached_data, name='append_to_cached_data'),
    url(r'^predict/$', views.predict_data, name='predict_data'),
    url(r'^check/server_status/$', views.check_server_status, name='check_server_status'),
    url(r'^test/hit_rate/$', views.test_hit_rate, name='test_hit_rate'),    
    url(r'^create/backup/$', views.create_backup, name='create_backup'),
    url(r'^use/backup/$', views.use_backup_data_as_main_data, name='use_backup_data_as_main_data'),
    url(r'^test/trained_model/$', views.test_current_trained_model, name='test_current_trained_model'),
    url(r'^test/case/$', views.test_case, name='testCase'),
    url(r'^list/backup/$', views.list_backups, name='list_backups'),    
]

urlpatterns = format_suffix_patterns(urlpatterns)