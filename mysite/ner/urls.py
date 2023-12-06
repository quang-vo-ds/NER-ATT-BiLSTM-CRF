from django.urls import path, re_path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    re_path(r'^get_tags/$', views.get_tags, name='get_tags'),
]