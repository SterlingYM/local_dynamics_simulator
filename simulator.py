# Scalar Field Simulator
# Yukei Murakami, sterling.astro@berkeley.edu
# version 0.2.0
# last updated: 12/2/2019

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bokeh.plotting import figure, output_notebook, output_file, show
from bokeh.layouts import column, row,layout
from bokeh.models import Button,CustomJS,TextInput,RadioButtonGroup,\
        ColumnDataSource,Legend,LegendItem,Toggle
from bokeh.models.widgets import Tabs, Panel
from bokeh.models.renderers import GlyphRenderer
from bokeh.palettes import viridis
from bokeh.io import curdoc

from scipy.constants import c as SOL, G, physical_constants
from astropy import units as u

############ constants #############
# TODO: check if these units are correct
G = G*u.meter**3/u.kg/u.second**2
SOL = SOL * (u.meter/u.second)
H0 = 70 *u.km/u.second/u.Mpc
rho_c = 3*(H0**2)/(8*np.pi*(G))

# set Lambda by fixing the time unit
t_unit = (1e6 * u.year).to(u.second) * SOL  # sec * SOL
Lambda = 1/t_unit

# length rescaling: r_tilde = r_actual/r_unit
r_unit = (1/Lambda).to(u.kpc)

# density rescaling: rho_tilde = rho_actual/rho_unit
rho_unit = (1/(3/rho_c*((H0/SOL)/Lambda)**2))

# mass rescaling: M_tilde = M_actual/M_unit (assuming Gaussian dist.)
# R_mass = 1e-2*r_unit
# M_unit = (rho_unit * R_mass**3 * np.pi**(3/2)).to(u.solMass)
kg_per_meter = SOL**2 /G


R_s = 1 #2 * G * rho_1 / SOL**2
L_0 = 1 #SOL / np.sqrt(G*rho_0)
eps = 1 #R_s / L_0


########### parameters ##############
# plot parameters
num_plot   = 10  # can be changed in GUI
num_legend = 10


def initialize():
    global N,L,dt,x_axis,R_mass,M,rho_1,rho_0,rho_J,dot_0,phi_0_array,dot_0_array,y0
    # Scaling params
    N  = 10000 # number of points
    L  = 1   # size of the box
    dt = 1e-4#0.01*L/N
    x_axis = np.linspace(0,L,N)

    # Matter params
    R_mass = 1e-2
    M = 1
    rho_1  = M/(R_mass**3 * np.pi**(3/2))
    rho_0 = (0.3*rho_c/rho_unit).to(u.km/u.km).value
    rho_J  = rho_1 * np.exp(-(x_axis/R_mass)**2) + rho_0

    # Initial Field params
    phi_0 = 0
    dot_0 = (H0*(1/(SOL))/Lambda).to(u.km/u.km).value #1e-5
    phi_0_array = np.full(N,phi_0)
    dot_0_array = np.full(N,dot_0)
    y0 = [phi_0_array,dot_0_array]
    return N,L,dt,x_axis,R_mass,M,rho_1,rho_0,rho_J,phi_0_array,dot_0_array,y0



############ coupling model #############
def calc_C(phi,C0,c_th,model='POWER'):
    if model == 'EXPON':
        C = 1+ C0*np.exp(c_th*phi)
        Cp = c_th * C0*np.exp(c_th*phi)
    elif model == 'POWER':
        C  = 1 + C0 * phi**c_th
        Cp = 0 if c_th==0 else C0* c_th * phi**(c_th-1)
    return C, Cp
    
def calc_D(phi,D0,d_th,model='POWER'):
    if model == 'EXPON':
        D  = D0 * np.exp(d_th*phi)
        Dp = d_th * D
    elif model == 'POWER':
        D  = D0 * phi**d_th
        Dp = 0 if d_th==0 else D0 * d_th * phi**(d_th-1)
    return D, Dp

############ Rescaled Field Equation ############
def calc_grad(phi,x_axis):
    norm = (1.-np.arange(len(phi))/float(N))**2/(x_axis[1]-x_axis[0])
    grad = np.zeros(phi.shape)
    grad[1:-1] = 0.5*(phi[2:]-phi[:-2])
    grad[0] = 0  # boundary
    grad[-1] = 0 # boundary
    return grad * norm

def calc_lap(phi,x_axis):
    norm = (1.-np.arange(len(phi))/float(N))**2/(x_axis[1]-x_axis[0])
    lap = np.zeros(phi.shape)
    lap[1:-1] = (phi[2:] + phi[:-2] - 2.*phi[1:-1]) + (phi[2:]-phi[:-2])/(np.arange(len(phi)-2)+1)
    lap[0] = (phi[1] - 1.*phi[0])    # boundary
    lap[-1] = (phi[-2] - 1.*phi[-1]) # boundary
    return lap*(norm**2)

def calc_deriv(y,eqn_params):
    # unpacking
    phi,dot = y
    if len(eqn_params)==4:
        eqn_params.append('POWER')
    C0,c_th,D0,d_th,model = eqn_params

    # quantities
    C,Cp = calc_C(phi,C0,c_th,model)
    D,Dp = calc_D(phi,D0,d_th,model)
    grad = calc_grad(phi,x_axis)
    lap = calc_lap(phi,x_axis)
    X = 0.5*(dot**2 - grad**2)
    rho_E = np.zeros(rho_J.shape)
    rho_E[(1.-2.*D*X/C)>=0] = (rho_J * C**3)[(1-2*D*X/C)>=0] * np.sqrt((1.-2.*D*X/C)[(1.-2.*D*X/C)>=0])

    # equation terms
    mathcalD = (C - D*(dot**2 - grad**2))
    Q1 = lap * mathcalD
    Q2 = rho_E * (Cp/2. + ((Cp/C)*D - Dp/2.)*(dot**2))
    Q3 = mathcalD + D*rho_E
    
    # check conditions
    ## TODO: avoid using global variable
    if np.any(C<0) or np.any(Q3<0):
        cauthy_sanity = False
        return 1;
    else:
        cauthy_sanity = True
        dot = dot
        dot2 = (Q1 + Q2) / Q3
        derivs = [dot,dot2]
        return np.array(derivs)



############ Evolution methods ################
def get_next_Euler(val_old,dt,eqn_params):
    val_old = np.array(val_old)
    return val_old + dt * calc_deriv(val_old,eqn_params)

def get_next_RK4(val_old,dt,eqn_params,return_D=False):
    val_old = np.array(val_old)
    k1 = val_old + 0.5*dt*calc_deriv(val_old,eqn_params)
    k2 = val_old + 0.5*dt*calc_deriv(k1,eqn_params)
    k3 = val_old +     dt*calc_deriv(k2,eqn_params)
    k4 =               dt*calc_deriv(k3,eqn_params)    
    return (k1 + 2*k2 + k3)/3 + k4/6 - val_old/3

############# simulator (no gui) #############
def simulate(C0,c_th,D0,d_th,ti,tf,dt,save_interval=0.5,
             method='RK4',model='POWER'):
    # initialize
    N,L,dt,x_axis,R_mass,M,rho_1,rho_0,rho_J,phi_0_array,dot_0_array,y0 = initialize()
    eqn_params = [C0,c_th,D0,d_th,model]
    val = y0

    # initial values
    t = 0
    cauthy_sanity = True
    phi_list = [phi_0_array]
    dot_list = [dot_0_array]
    t_list   = [t]

    ## main loop
    t_precision = int(abs(np.log10(dt))+1)
    save_counter = 0
    while (t<=tf and cauthy_sanity):
        if (t%save_interval <= 10**(-t_precision)) or \
        (abs(t%save_interval-save_interval) <= 10**(-t_precision)) : #save_interval:
            print(f'\rSimulating... t = {t:.1f}/{tf}',end='')
        if t==0 or t==dt/2:
            sim_dt = dt/2
        else:
            sim_dt = dt
        t = np.around(t+sim_dt,t_precision)

        if method=='Euler':
            phi_new, dot_new = get_next_Euler(val,sim_dt,eqn_params)
        if method=='RK4':
            phi_new, dot_new = get_next_RK4(val,sim_dt,eqn_params)
            
        if (t%save_interval <= 10**(-t_precision)) or \
        (abs(t%save_interval-save_interval) <= 10**(-t_precision)) : #save_interval:
            t_list.append(t)
            phi_list.append(phi_new)
            dot_list.append(dot_new)
        val = [phi_new,dot_new]
        
    t_list = np.array(t_list)
    phi_list = np.array(phi_list)
    dot_list = np.array(dot_list)
    return phi_list,dot_list,t_list

def plot(x_axis,phi,t,N_plots=10,fig=None,ax=None,cmap='winter_r',figsize=(8,6),
         xlim=(1e-4,1),xlabel=r'$\tilde{r}$',ylabel=r'$\tilde\phi$',yscale='linear',return_axes=False,
         return_norm=False,add_colorbar=True):
    if ax == None and fig==None:
        fig,ax = plt.subplots(1,1,figsize=figsize,dpi=100)
    elif ax==None or fig==None:
        raise ValueError('When axis is specified, both fig and ax need to be provided.')
    cm = plt.get_cmap(cmap)
    for j,i in enumerate(range(0,len(t),int((len(t)-1)/N_plots))):
        ax.plot(x_axis,phi[i],c=cm(j/N_plots))
    ax.set_xlim(xlim)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    
    # colorbar
    norm = mpl.colors.Normalize(vmin=t[0], vmax=np.round(t[-1]))
    if add_colorbar:
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cm, norm=norm,orientation='vertical')
        cb1.ax.set_title(r'$\tilde{t}$')
        fig.add_axes(ax_cb)
    if return_norm:
        return norm
        
    if return_axes:
        return ax,ax_cb
    
def calc_Drho(phi,dot,eqn_params):
    C0,c_th,D0,d_th,model = eqn_params

    # quantities
    C,Cp = calc_C(phi,C0,c_th,model)
    D,Dp = calc_D(phi,D0,d_th,model)
    grad = calc_grad(phi,x_axis)
    X = 0.5*(dot**2 - grad**2)
    rho_E = np.zeros(rho_J.shape)
    rho_E[(1.-2.*D*X/C)>=0] = (rho_J * C**3)[(1-2*D*X/C)>=0] * np.sqrt((1.-2.*D*X/C)[(1.-2.*D*X/C)>=0])
    return D*rho_E
    
def calc_mathcalD(phi,dot,eqn_params):
    C0,c_th,D0,d_th,model = eqn_params

    # quantities
    C,Cp = calc_C(phi,C0,c_th,model)
    D,Dp = calc_D(phi,D0,d_th,model)
    grad = calc_grad(phi,x_axis)
    X = 0.5*(dot**2 - grad**2)
    rho_E = np.zeros(rho_J.shape)
    rho_E[(1.-2.*D*X/C)>=0] = (rho_J * C**3)[(1-2*D*X/C)>=0] * np.sqrt((1.-2.*D*X/C)[(1.-2.*D*X/C)>=0])

    # equation terms
    mathcalD = (C - D*(dot**2 - grad**2))
    return mathcalD

############ plotter ###############
def simulate_fancy(source_phi,source_dot,C0,c_th,D0,d_th,ti,tf,dt,\
        fig_list,num_plot=10,title="No Title",method='RK4'):
    # initialize
    N,L,dt,x_axis,R_mass,M,rho_1,rho_0,rho_J,phi_0_array,dot_0_array,y0 = initialize()
    p1,p2,p3,p4     = fig_list
    eqn_params      = [C0,c_th,D0,d_th,'POWER']
    plot_interval   = (tf-ti) /num_plot
    val = y0
    print('Eqn_params:',eqn_params)

    # prepare color map
    cm = []
    cm_LG = []
    plt_cm = plt.get_cmap('winter')
    for i in range(num_plot+1):
        r,g,b,_ = 255*np.array(plt_cm(i/(num_plot)))
        cm.append("#%02x%02x%02x" % (int(r), int(g), int(b)))

    # initial values
    cauthy_sanity = True
    t = 0
    source_phi.stream({'x':[list(x_axis)],
                       'y':[list(phi_0_array)],
                       'labels':['t = 0'],
                       'color':[cm[int(0/tf*num_plot)]]})
    source_dot.stream({'x':[list(x_axis)],
                       'y':[list(dot_0_array)],
                       'labels':['t = 0'],
                       'color':[cm[int(0/tf*num_plot)]]})

    ## plot initial values
    t_precision = 10**(-1*int(abs(np.log10(dt))+1))
    phi_LG = []
    dot_LG = []
    t_list   = []
    
    print(f'\rSimulating... t = {t:.1f}/{tf}',end='')
    while (t<=tf and cauthy_sanity):    
        if t==0 or t==dt/2:
            sim_dt = dt/2
        else:
            sim_dt = dt
        t = np.around(t+sim_dt,int(abs(np.log10(dt))+1))
        if method=='Euler':
            phi_new, dot_new = get_next_Euler(val,sim_dt,eqn_params)
        if method=='RK4':
            phi_new, dot_new = get_next_RK4(val,sim_dt,eqn_params)

        # save occasionally
        shouldSave = (t%plot_interval <= t_precision) or \
        (abs(t%plot_interval-plot_interval) <= t_precision)
        if shouldSave:
            print(f'\rSimulating... t = {t:.1f}/{tf}',end='')
            label = 't = {:.2f}'.format(t)
            source_phi.stream({'x':[list(x_axis)],
                               'y':[list(phi_new)],
                               'labels':[label],
                               'color':[cm[int(t/tf*num_plot)]]})
            source_dot.stream({'x':[list(x_axis)],
                               'y':[list(dot_new)],
                               'labels':[label],
                               'color':[cm[int(t/tf*num_plot)]]})
        phi_LG.append('' if np.isnan(phi_new[0]/phi_new[-1]) else phi_new[0]/phi_new[-1])
        dot_LG.append('' if np.isnan(dot_new[0]/dot_new[-1]) else dot_new[0]/dot_new[-1])
        t_list.append(t)
        val = [phi_new,dot_new]
    
    # LG plot
    # TODO: change this to the ColumnDataSource so that all-reset function can be added
    for j in range(len(phi_LG)):
        r,g,b,_ = 255*np.array(plt_cm(j/len(phi_LG)))
        cm_LG.append("#%02x%02x%02x" % (int(r), int(g), int(b)))
    p3.line(t_list,phi_LG)
    p3.circle(t_list,phi_LG,size=1,color=cm_LG,legend_label=title)
    p4.line(t_list,dot_LG)
    p4.circle(t_list,dot_LG,size=1,color=cm_LG,legend_label=title)
    
    # phi & dot plot
    r1 = p1.multi_line('x','y',legend_field='labels',line_color='color',\
            line_width=1.5,source=source_phi)
    r2 = p2.multi_line('x','y',legend_field='labels',line_color='color',\
            line_width=1.5,source=source_dot)
    
    # formatting
    #####################################
    ## TODO: make legends individual
    ##legend_list_1 = []
    ##legend_list_2 = []
    ##for i in range(num_plot):
    ##    legend_list_1.append(LegendItem(label=source_phi.data['labels'][i],\
    ##       renderers=[r1],index=i))
    ##    legend_list_2.append(LegendItem(label=source_dot.data['labels'][i],\
    ##       renderers=[r2],index=i))
    ##p1.add_layout(Legend(items=legend_list_1),'right')
    ##p2.add_layout(Legend(items=legend_list_2))
    #####################################
    p1.legend.click_policy="hide"
    p2.legend.click_policy="hide"
    p3.legend.click_policy="hide"
    p4.legend.click_policy="hide"
    
    for p in [p1,p2]:
        p.x_range.start = x_axis[1]
        p.x_range.end   = x_axis[-1]
#     for p in [p1,p2,p3,p4]:
#         p.x_range.start = None
#         p.x_range.end   = None
#         p.y_range.start = None
#         p.y_range.end   = None

    print('\nSimulation Completed')

########### wrapping GUI ###########
def start_gui():
    #### internally called functions ####
    def generate_figs():
        p1 = figure(plot_width=1200,plot_height=900, title="phi", x_axis_label='r',x_axis_type='log')
        p2 = figure(plot_width=1200,plot_height=900, title="dot", x_axis_label='r',x_axis_type='log')
        p3 = figure(plot_width=1200,plot_height=900, title="phi Local / Global", x_axis_label='t')
        p4 = figure(plot_width=1200,plot_height=900, title="dot Local / Global", x_axis_label='t') 
        p1.sizing_mode = 'stretch_both'
        p2.sizing_mode = 'stretch_both'
        p3.sizing_mode = 'stretch_both'
        p4.sizing_mode = 'stretch_both'
        # TODO: make plot width responsive
        return p1,p2,p3,p4
    def switch_plot():
        plot_shown.children[0] = fig_list[plot_selector.active]
    def refresh_plot(): 
        # temporary bug fix: newly generated plot is broken without this
        plot_shown.children[0] = dummy_fig
        plot_shown.children[0] = fig_list[plot_selector.active]
    def start_click():
        print('Run clicked, simulating...')
        title = title_input.value
        C0   = float(C0_input.value)
        D0   = float(D0_input.value)
        c_th = float(c_th_input.value)
        d_th = float(d_th_input.value)
        ti   = float(ti_input.value)
        tf   = float(tf_input.value)
        dt   = float(dt_input.value)
        N_plot = int(N_plot_input.value)
        simulate_fancy(source_phi,source_dot,\
                C0=C0,c_th=c_th,D0=D0,d_th=d_th,ti=ti,tf=tf,fig_list=fig_list,\
                dt=dt,num_plot=N_plot,title=title,method='RK4')
        refresh_plot()
    def clear_plot():
        source_phi.data = {k: [] for k in source_phi.data}
        source_dot.data = {k: [] for k in source_dot.data}
        refresh_plot()
    def reset_fig():
        return 0
        # TODO: add a function to reset without restarting the window
        #fig_list = generate_figs()
        #p1,p2,p3,p4 = fig_list
    def legend_showhide():
        Labels = ['Hide Legend','Show Legend']
        status = legend_switch.active
        print(status)
        legend_switch.label=Labels[status]
        for p in fig_list:
            p.legend.visible=False if status else True
    #### generate gui ####
    # figures (tabs)
    dummy_fig   = figure(plot_width=1200,plot_height=900,sizing_mode='stretch_both')
    fig_list    = generate_figs()
    p1,p2,p3,p4 = fig_list
    plot_shown  = row(children=[p1],sizing_mode='stretch_both')
    # data for phi & dot
    source_phi  = ColumnDataSource(data=dict(x=[], y=[], labels=[], color=[]))
    source_dot  = ColumnDataSource(data=dict(x=[], y=[], labels=[], color=[]))
    # User inputs
    title_input  = TextInput(value="Title",title="Title",sizing_mode='stretch_both')
    C0_input     = TextInput(value="1e-2",title="C0",sizing_mode='stretch_both')
    D0_input     = TextInput(value="0",title="D0",sizing_mode='stretch_both')
    c_th_input   = TextInput(value="1",title="c_th",sizing_mode='stretch_both')
    d_th_input   = TextInput(value="0",title="d_th",sizing_mode='stretch_both')
    ti_input     = TextInput(value="0",title="ti",sizing_mode='stretch_both')
    tf_input     = TextInput(value="1",title="tf",sizing_mode='stretch_both')
    dt_input     = TextInput(value="0.0001",title="dt",sizing_mode='stretch_both')
    N_plot_input = TextInput(value='10',title='Number of Plotted Lines',sizing_mode='stretch_both')
    # button, selector
    start_button  = Button(label='Run',button_type="success",sizing_mode='scale_both')
    clear_button  = Button(label='Clear',button_type='warning',sizing_mode='scale_both')
    legend_switch = Toggle(label='Hide Legend',button_type='primary',sizing_mode='scale_both')
    #reset_button  = Button(label='Reset All',button_type='warning',sizing_mode='scale_both')
    method_switch = RadioButtonGroup(labels=["RK4","Euler"],active=0,sizing_mode='scale_both')
    plot_selector = RadioButtonGroup(\
            labels=["Field (phi)","Velocity (dot)","Field Local/Global","Velocity Local/Global"],\
            active=0,sizing_mode='scale_both')
    # button behavior
    start_button.on_click(start_click)
    clear_button.on_click(clear_plot)
    legend_switch.on_click(lambda new: legend_showhide())
    #reset_button.on_click(reset_fig)
    plot_selector.on_change('active',lambda attr,old,new: switch_plot())
    # layout
    cols = column([
        row(title_input,sizing_mode='scale_width'),
        row([C0_input,c_th_input],sizing_mode='scale_width'),
        row([D0_input,d_th_input],sizing_mode='scale_width'),
        row([ti_input,tf_input],sizing_mode='scale_width'),
        row([dt_input,N_plot_input],sizing_mode='scale_width'),
        row([method_switch,clear_button,legend_switch,start_button],sizing_mode='scale_width'),
        row([plot_selector],sizing_mode='scale_width'),
        plot_shown],
        sizing_mode='stretch_both')
    curdoc().add_root(cols)
    
def start_jupyter(doc):
    #### internally called functions ####
    def generate_figs():
        p1 = figure(plot_width=1200,plot_height=600, title="phi", x_axis_label='r',x_axis_type='log')
        p2 = figure(plot_width=1200,plot_height=600, title="dot", x_axis_label='r',x_axis_type='log')
        p3 = figure(plot_width=1200,plot_height=600, title="phi Local / Global", x_axis_label='t')
        p4 = figure(plot_width=1200,plot_height=600, title="dot Local / Global", x_axis_label='t') 
        p1.sizing_mode = "scale_width"
        p2.sizing_mode = "scale_width"
        p3.sizing_mode = "scale_width"
        p4.sizing_mode = "scale_width"
        # TODO: make plot width responsive
        return p1,p2,p3,p4
    def switch_plot():
        plot_shown.children[0] = fig_list[plot_selector.active]
    def refresh_plot(): 
        # temporary bug fix: newly generated plot is broken without this
        plot_shown.children[0] = dummy_fig
        plot_shown.children[0] = fig_list[plot_selector.active]
    def start_click():
        print('Run clicked, simulating...')
        title = title_input.value
        C0   = float(C0_input.value)
        D0   = float(D0_input.value)
        c_th = float(c_th_input.value)
        d_th = float(d_th_input.value)
        ti   = float(ti_input.value)
        tf   = float(tf_input.value)
        dt   = float(dt_input.value)
        N_plot = int(N_plot_input.value)
        simulate_fancy(source_phi,source_dot,\
                C0=C0,c_th=c_th,D0=D0,d_th=d_th,ti=ti,tf=tf,fig_list=fig_list,\
                dt=dt,num_plot=N_plot,title=title,method='RK4')
#         refresh_plot()
    def clear_plot():
        source_phi.data = {k: [] for k in source_phi.data}
        source_dot.data = {k: [] for k in source_dot.data}
        refresh_plot()
    def reset_fig():
        return 0
        # TODO: add a function to reset without restarting the window
        #fig_list = generate_figs()
        #p1,p2,p3,p4 = fig_list
    def legend_showhide():
        Labels = ['Hide Legend','Show Legend']
        status = legend_switch.active
        legend_switch.label=Labels[status]
        for p in fig_list:
            p.legend.visible=False if status else True
    #### generate gui ####
    # figures (tabs)
    dummy_fig   = figure(plot_width=1200,plot_height=900,sizing_mode='stretch_both')
    fig_list    = generate_figs()
    p1,p2,p3,p4 = fig_list
    plot_shown  = row(children=[p1],sizing_mode='scale_width')
    # data for phi & dot
    source_phi  = ColumnDataSource(data=dict(x=[], y=[], labels=[], color=[]))
    source_dot  = ColumnDataSource(data=dict(x=[], y=[], labels=[], color=[]))
    # User inputs
    title_input  = TextInput(value="Title",title="Title",sizing_mode='stretch_both')
    C0_input     = TextInput(value="1e-2",title="C0",sizing_mode='stretch_both')
    D0_input     = TextInput(value="0",title="D0",sizing_mode='stretch_both')
    c_th_input   = TextInput(value="1",title="c_th",sizing_mode='stretch_both')
    d_th_input   = TextInput(value="0",title="d_th",sizing_mode='stretch_both')
    ti_input     = TextInput(value="0",title="ti",sizing_mode='stretch_both')
    tf_input     = TextInput(value="1",title="tf",sizing_mode='stretch_both')
    dt_input     = TextInput(value="0.0001",title="dt",sizing_mode='stretch_both')
    N_plot_input = TextInput(value='10',title='Number of Plotted Lines',sizing_mode='stretch_both')
    # button, selector
    start_button  = Button(label='Run',button_type="success",sizing_mode='scale_both')
    clear_button  = Button(label='Clear',button_type='warning',sizing_mode='scale_both')
    legend_switch = Toggle(label='Hide Legend',button_type='primary',sizing_mode='scale_both')
    #reset_button  = Button(label='Reset All',button_type='warning',sizing_mode='scale_both')
    method_switch = RadioButtonGroup(labels=["RK4","Euler"],active=0,sizing_mode='scale_both')
    plot_selector = RadioButtonGroup(\
            labels=["Field (phi)","Velocity (dot)","Field Local/Global","Velocity Local/Global"],\
            active=0,sizing_mode='scale_both')
    # button behavior
    start_button.on_click(start_click)
    clear_button.on_click(clear_plot)
    legend_switch.on_click(lambda new: legend_showhide())
    #reset_button.on_click(reset_fig)
    plot_selector.on_change('active',lambda attr,old,new: switch_plot())
    # layout
    cols = column([
        row(title_input,sizing_mode='scale_width'),
        row([C0_input,c_th_input],sizing_mode='scale_width'),
        row([D0_input,d_th_input],sizing_mode='scale_width'),
        row([ti_input,tf_input],sizing_mode='scale_width'),
        row([dt_input,N_plot_input],sizing_mode='scale_width'),
        row([method_switch,clear_button,legend_switch,start_button],sizing_mode='scale_width'),
        row([plot_selector],sizing_mode='scale_width'),
        plot_shown],
        sizing_mode='stretch_both')
    doc.add_root(cols)
    
# if __name__=="__main__":
start_gui()
