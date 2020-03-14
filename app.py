# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:58:46 2020

@author: joeda
"""

import os
from random import randint
import numpy as np
import flask
from numba import njit

import plotly.express as px
#@njit(cache=True)
def diffusif_leapfrog(c,N,D,dt,dx):

    return D*dt/dx**2*((c[1:N+1,2:N+2] + c[2:N+2,1:N+1] - 4*c[1:N+1,1:N+1] + c[1:N+1,0:N] + c[0:N,1:N+1]))

#@njit(cache=True)
def  Energie4(spin,N,T,H,J,couleur) :


        #Calcul E et Eprime
        up=spin[1:N+1,2:N+2]

        down=spin[1:N+1,0:N]
        left=spin[0:N,1:N+1]
        right=spin[2:N+2,1:N+1]
        E=-J*spin[1:N+1,1:N+1]*(up+down+left+right)-H*spin[1:N+1,1:N+1]
        #gpu(spin,H,N,E)

        p=np.exp((2*E)/T[1:N+1,1:N+1])


        p2=np.reshape(np.random.rand(N**2),(N,N))
        p3=np.where(np.logical_and(p2<p,couleur==1),-1,1)
        spin[1:N+1,1:N+1]=spin[1:N+1,1:N+1]*p3



        return spin,E

#@njit(cache=True)
def  Energie8(spin,N,T,H,J,couleur) :


        #Calcul E et Eprime
        #premier voisins
        up=spin[1:N+1,2:N+2]
        down=spin[1:N+1,0:N]
        left=spin[0:N,1:N+1]
        right=spin[2:N+2,1:N+1]
        premiervoisins=2*(up+down+left+right)

        #coins
        upd=spin[2:N+2,2:N+2] #en haut a droite
        upg=spin[0:N,2:N+2] #en haut a gauche
        downd=spin[2:N+2,0:N]
        downg=spin[0:N,0:N]
        secondvoisins=(upd+downd+upg+downg)
        E=-J*spin[1:N+1,1:N+1]*(premiervoisins+secondvoisins)-H*spin[1:N+1,1:N+1]
        #gpu(spin,H,N,E)

        p=np.exp((2*E)/T[1:N+1,1:N+1])


        p2=np.reshape(np.random.rand(N**2),(N,N))
        p3=np.where(np.logical_and(p2<p,couleur==1),-1,1)
        spin[1:N+1,1:N+1]*=p3



        return spin,E

#@njit(cache=True)
def  Energie6(spin,N,T,H,J,couleur) :



        #Calcul E et Eprime
        #premier voisins
        up=spin[1:N+1,2:N+2]
        down=spin[1:N+1,0:N]
        left=spin[0:N,1:N+1]
        right=spin[2:N+2,1:N+1]
        premiervoisins=(up+down+left+right)

        #coin
        downd=spin[2:N+2,0:N]
        downg=spin[0:N,0:N]
        secondvoisins=(downd+downg)
        E=-J*spin[1:N+1,1:N+1]*(premiervoisins+secondvoisins)-H*spin[1:N+1,1:N+1]
        #gpu(spin,H,N,E)
        p=np.exp((2*E)/T[1:N+1,1:N+1])


        p2=np.reshape(np.random.rand(N**2),(N,N))
        p3=np.where(np.logical_and(p2<p,couleur==1),-1,1)
        spin[1:N+1,1:N+1]*=p3



        return spin,E


#@njit(cache=True)
def periodic(f,N):

    #périodicite horizontale et verticale
    f[1:N+1,0]=f[1:N+1,N]
    f[1:N+1,N+1]=f[1:N+1,1]
    f[0,1:N+1]= f[N,1:N+1]
    f[N+1,1:N+1]=f[1,1:N+1]

    #coins

    f[0,0]=f[N-1,N-1]
    f[N+1,N+1]=f[2,2]
    f[0,N+1]=f[N-1,2]
    f[N+1,0]=f[2,N-1]

    return f


#@njit(cache=True)
def indice4(N) :
    spin=np.zeros((N+2,N+2))+1 #spin initial
     #Tableau d'indice
    spinNoir = np.ones((N+2,N+2))
    spinNoir[::2,::2] = 0
    spinNoir[1::2,1::2] = 0
    #spinNoir=np.where(spinNoir==0)
    #spinBlanc=np.where(spinNoir==0)
    spinBlanc = np.ones((N+2,N+2))
    spinBlanc[1::2,::2] = 0
    spinBlanc[::2,1::2] = 0
    spinBlanc=spinBlanc[1:N+1,1:N+1]
    spinNoir=spinNoir[1:N+1,1:N+1]
    return spin,[spinNoir,spinBlanc]
#@njit(cache=True)
def indice8(N) :
    spin=np.zeros((N+2,N+2))+1 #spin initial
     #Tableau d'indice
    spinbleu = np.zeros((N+2,N+2),dtype=np.bool)
    spinbleu[::2,::2] = 1

    spinvert= np.zeros((N+2,N+2),dtype=np.bool)
    spinvert[1::2,::2]=1

    spinjaune= np.zeros((N+2,N+2),dtype=np.bool)
    spinjaune[1::2,1::2]=1

    spinorange= np.zeros((N+2,N+2),dtype=np.bool)
    spinorange[::2,1::2]=1


    spinbleu=spinbleu[1:N+1,1:N+1]
    spinvert=spinvert[1:N+1,1:N+1]
    spinjaune=spinjaune[1:N+1,1:N+1]
    spinorange=spinorange[1:N+1,1:N+1]
    return spin,[spinbleu,spinvert,spinjaune,spinorange]


#---------------------------------------------------
#@njit
def main(H0,T0,R,t0,Tmax,sigma,D,nIter,stencil) :
    #Définition des paramètres globaux
    H0=H0/10
    T0/=10
    D=D/100
    Tmax/=100
    #filtered_df = df[df.year == selected_year]
    dT=Tmax-T0
    N=64



    #Déclaration des tableaux
    #spin=-1+2*np.random.randint(low=0,high=2,size=(N+2,N+2)) #spin aléatoire initial [-1,1]
    if stencil=='4' :
        spin,couleur=indice4(N)
        J=1
        Energie=Energie4

    elif stencil=='8' :
        spin,couleur=indice8(N)
        Energie=Energie8
        J=1/3

    elif stencil=='6':
        spin,couleur=indice8(N)
        Energie=Energie6
        J=2/3


    #Calcul préalable de la région chauffée
    x0=np.int(N/2) # on centre notre laser
    y0=x0
    f=np.zeros((N,N),dtype=np.int8)
    for i in range(0,N) :
        for j in range(0,N) :
                r=(j-y0)**2+(i-x0)**2
                if r<=R**2 :
                    f[i][j]=1


    #Champs magnétique et Température
    T=np.zeros((N+2,N+2),dtype=np.float32)+T0
    H=H0
    #Itération temporelle
    for i in range(0,nIter) :
        #H=H0*np.sin(2*pi*nIter/5)

        Tsource=dT*f*np.exp(-((i-t0)/sigma)**2)

        T[1:N+1,1:N+1]+=diffusif_leapfrog(T,N,D,1,1)+Tsource

        #Vrm pas sure pour Eprime
        #operation sur les blancs

         #operation sur les différentes couleur
        for color in couleur :
            spin,E=Energie(spin,N,T,H,J,color)





        #print(E_mean)
        #print(E)
        #print(spin[1:N+1,1:N+1])


        #Animation.append([plt.imshow(spin, animated=True)])
        spin=periodic(spin,N)
        T=periodic(T,N)
        #if i%10==0 :
           # plt.imshow(spin)
           # plt.show()

    return spin

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = flask.Flask(__name__)

server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}


fig1 = px.imshow(np.zeros((66,66)))
fig1.update_layout(width=700, height=700)
app.layout = html.Div(children=[
    html.H1('Simulation de gravure magnétique au laser \n à l\'aide du modèle d\'Ising'),
    dcc.Graph(
        id='graph-with-slider',
        animate=False,
        figure=fig1
    ),

         html.Label('Stencil'),
    html.Div(dcc.Dropdown(
    options=[
        {'label': 'Carre 8', 'value': '8'},
        {'label': 'triangulaire 6', 'value': '6'},
        {'label': 'closest 4', 'value': '4'}
    ],
    searchable=True,
    id='stencil',
    value='4',
    placeholder="Stencil"),  style= {'padding': 20}),


        html.Label('Champs magnétique en mT'),
    html.Div(dcc.Slider(
        id='H0',
        min=0,
        max=20,
        marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0,20,2)},
        value=2,
    ), style= {'padding': 20}),

        html.Label('Température basale'),
    html.Div(dcc.Slider(
        id='T0',
        min=0,
        max=10,
        marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 10,1)},
        value=1,
        ), style= {'padding': 20}),


        html.Label('Rayon'),
    html.Div(dcc.Slider(
        id='R',
        min=0,
        max=20,
        marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 20,2)},
        value=10,
    ), style= {'padding': 20}),



         html.Label('Coefficient de chauffage du laser'),
    html.Div(dcc.Slider(
        id='Tmax',
        min=10,
        max=30,
        marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(10,30,2)},
        value=25,
    ), style= {'padding': 20}),

         html.Label('Durée du pulse laser (sigma)'),
    html.Div(dcc.Slider(
        id='sigma',
        min=0,
        max=50,
        marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0,50,5)},
        value=10,
    ), style= {'padding': 20}),

         html.Label('Coefficient de Diffusion D en mD'),
    html.Div(dcc.Slider(
        id='D',
        min=0,
        max=5,
        marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0,5,1)},
        value=1,
    ), style= {'padding': 20}),

         html.Label('Nombre d\'itération'),
    html.Div(dcc.Slider(
        id='nIter',
        min=100,
        max=1000,
        marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(100,1000,100)},
        value=200,
    ), style= {'padding': 20}),


        html.H1('Par Jonathan Beaulieu-Emond'),

])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('stencil', 'value'),
    Input('H0', 'value'),
    Input('T0', 'value'),
    Input('R', 'value'),
    Input('Tmax', 'value'),
    Input('sigma', 'value'),
    Input('D', 'value'),
    Input('nIter', 'value')])

def update_figure(stencil,H0,T0,R,Tmax,sigma,D,nIter):


    print('stencil=',stencil,type(stencil))
    spin= main(-H0,T0,R,100,Tmax,sigma,D,nIter,stencil)

    fig = px.imshow(spin, color_continuous_scale='gray')
    fig.update_layout(width=700, height=700)





    return fig



if __name__ == '__main__':
    app.run_server(debug=False)
