# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .models import Records
from django.shortcuts import render, render_to_response
# Create your views here.
def index(request):
    records = Records.objects.all()[:10]    #getting the first 10 records
    context = {
        'records': records
    }
    return render(request, 'records.html', context)

def details(request, id):
    record = Records.objects.get(id=id)
    print record
    #context = {
    #    'record' : record
    #}
    return render(request,'details.html', {'record':record})
