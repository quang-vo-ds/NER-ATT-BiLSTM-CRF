from django.shortcuts import render

# Create your views here.
import json
from django.contrib.auth.models import User #####
from django.http import JsonResponse , HttpResponse ####


from .run_tagger import *
predictor = TaggerPredictor(gpu = -1,
                            load = 'src/pretrained/2023_12_02_02-57_22_tagger.hdf5')

def index(request):
    return HttpResponse("Hello, world")


# https://pypi.org/project/wikipedia/#description
def get_tags(request):
    input_sentence = request.GET.get('input_sentence', None)

    print('input_sentence:', input_sentence)

    data = {
        'tag': predictor.predict(input_sentence),
        'raw': 'Successful',
    }

    print('json-data to be sent: ', data)

    return JsonResponse(data)