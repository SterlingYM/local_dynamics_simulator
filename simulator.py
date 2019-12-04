# Scalar Field Simulator
# Yukei Murakami, sterling.astro@berkeley.edu
# version 0.2.0
# last updated: 12/2/2019

import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, output_file, show
from bokeh.layouts import column, row,layout
from bokeh.models import Button,CustomJS,TextInput,RadioButtonGroup,\
        ColumnDataSource,Legend,LegendItem,Toggle
from bokeh.models.widgets import Tabs, Panel
from bokeh.models.renderers import GlyphRenderer
from bokeh.palettes import viridis
from bokeh.io import curdoc

########### parameters ##############
# Scaling params
resolution = 100
x_axis = np.logspace(0,3,resolution)

# Matter params
R_mass = 5e0
M      = 1e30
rho_1  = 4*M/(R_mass**3 * np.sqrt(np.pi))
rho_0 = 1
rho = rho_1 / np.sqrt(2 * np.pi * R_mass**2) * np.exp(-x_axis**2 / (2* R_mass**2)) + rho_0

# Initial Field params
phi_0 = 0 #0.1
dot_0 = 0.1
phi_0_array = np.full(resolution,phi_0)
dot_0_array = np.full(resolution,dot_0)
y0 = [phi_0_array,dot_0_array]

# plot parameters
num_plot   = 10  # can be changed in GUI
num_legend = 10


############ constants #############
SOL = 3.0 * 10**8
G   = 6.67 * 10**-11
R_s = 2 * G * rho_1 / SOL**2
L_0 = SOL / np.sqrt(G*rho_0)
eps = R_s / L_0



############ coupling model #############
def calc_C(phi,C0,c_th):
    return C0 * np.exp(c_th*phi)

def calc_D(phi,D0,d_th):
    return D0 * np.exp(d_th*phi)


############ Rescaled Quantities ############
def r_tilde(r):
    return r / R_s

def t_tilde(t):
    return t* SOL / R_s

def rho_tilde(rho):
    return rho / rho_0

def D_tilde(phi,D0,d_th):
    return calc_D(phi,D0,d_th) / (L_0**2)


############ Rescaled Field Equation ############
def calc_grad(phi):
    return np.gradient(phi,x_axis)

def calc_grad2(phi):
    return np.gradient(calc_grad(phi),x_axis)
# TODO: add the boundary condition

def return_val(y,eqn_params):
    # unpacking
    phi,dot = y
    C0,c_th,D0,d_th = eqn_params

    # rescale
    C = calc_C(phi,C0,c_th)
    D = D_tilde(phi,D0,d_th)
    grad = calc_grad(phi)
    grad2 = calc_grad2(phi)
    C_grad = c_th * C
    D_grad = d_th * D

    # equation quantities
    Q1 = (grad2 + (2/x_axis)*grad) * (C - (D/eps**2) * (dot**2 - grad**2))
    Q2 = rho_tilde(rho) * (eps**2*C_grad/2 + (c_th*D - D_grad/2)*dot**2)
    Q3 = C - (D/eps**2) * (dot**2 - grad**2) + D*rho

    # return values
    dot = dot
    dot2 = (Q1 + Q2) / Q3
    derivs = [dot,dot2]
    return np.array(derivs)


############ Simulator ################
def get_next_Euler(val_old,dt,eqn_params):
    val_old = np.array(val_old)
    return val_old + dt * return_val(val_old,eqn_params)


def get_next_RK4(val_old,dt,eqn_params):
    val_old = np.array(val_old)
    k1 = dt * return_val(val_old     ,eqn_params)
    k2 = dt * return_val(val_old+k1/2,eqn_params)
    k3 = dt * return_val(val_old+k2/2,eqn_params)
    k4 = dt * return_val(val_old+k3  ,eqn_params)
    return val_old + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

############ plotter ###############
def simulate_fancy(source_phi,source_dot,C0,c_th,D0,d_th,ti,tf,dt,\
        fig_list,num_plot=10,title="No Title",method='RK4'):
    # initialize
    p1,p2,p3,p4     = fig_list
    eqn_params      = [C0,c_th,D0,d_th]
    plot_interval   = (tf-ti)/dt /num_plot
    legend_interval = int(num_plot / num_legend)
    t   = np.linspace(ti,tf,int((tf-ti)/dt))
    val = y0

    # prepare color map
    cm = []
    cm_LG = []
    plt_cm = plt.get_cmap('winter')
    for i in range(num_plot+1):
        r,g,b,_ = 255*np.array(plt_cm(i/(num_plot)))
        cm.append("#%02x%02x%02x" % (int(r), int(g), int(b)))

    # initial values
    new_phi = {'x':[list(x_axis)],'y':[list(phi_0_array)],\
            'labels':['t = 0'],'color':[cm[int(0/tf*num_plot)]]}
    new_dot = {'x':[list(x_axis)],'y':[list(dot_0_array)],\
            'labels':['t = 0'],'color':[cm[int(0/tf*num_plot)]]}
    source_phi.stream(new_phi)
    source_dot.stream(new_dot)

    ## plot initial values
    phi_LG = []
    dot_LG = []
    plot_counter   = 0
    legend_counter = 0
    for time in t:
        if method=='Euler':
            phi_new, dot_new = get_next_Euler(val,dt,eqn_params)
        if method=='RK4':
            phi_new, dot_new = get_next_RK4(val,dt,eqn_params)
        if plot_counter >= plot_interval:
            ########################
            # TODO: reduce number of legends
            #if legend_counter >= legend_interval:
            #    label = ['t = {:.2f}'.format(time)]
            #    legend_counter = 0
            #else:
            #    label = []
            label = ['t = {:.2f}'.format(time)] # tmp
            #########################
            new_phi = {'x':[list(x_axis)],'y':[list(phi_new)],\
                    'labels':label,'color':[cm[int(time/tf*num_plot)]]}
            new_dot = {'x':[list(x_axis)],'y':[list(dot_new)],\
                    'labels':label,'color':[cm[int(time/tf*num_plot)]]}
            source_phi.stream(new_phi)
            source_dot.stream(new_dot)
            plot_counter = 0
            legend_counter += 1
        phi_LG.append('' if np.isnan(phi_new[0]/phi_new[-1]) else phi_new[0]/phi_new[-1])
        dot_LG.append('' if np.isnan(dot_new[0]/dot_new[-1]) else dot_new[0]/dot_new[-1])
        plot_counter   += 1
        val = [phi_new,dot_new]
    
    # LG plot
    for j in range(len(phi_LG)):
        r,g,b,_ = 255*np.array(plt_cm(j/len(phi_LG)))
        cm_LG.append("#%02x%02x%02x" % (int(r), int(g), int(b)))
    p3.circle(t,phi_LG,size=3,color=cm_LG,legend_label=title)
    p4.circle(t,dot_LG,size=3,color=cm_LG,legend_label=title)
    
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
    for p in [p1,p2,p3,p4]:
        p.x_range.start = None
        p.x_range.end   = None
        p.y_range.start = None
        p.y_range.end   = None


   
    print('Simulation Completed')

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
                dt=dt,num_plot=N_plot,title=title,method='Euler')
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
    C0_input     = TextInput(value="1e-30",title="C0",sizing_mode='stretch_both')
    D0_input     = TextInput(value="0",title="D0",sizing_mode='stretch_both')
    c_th_input   = TextInput(value="1",title="c_th",sizing_mode='stretch_both')
    d_th_input   = TextInput(value="0",title="d_th",sizing_mode='stretch_both')
    ti_input     = TextInput(value="0",title="ti",sizing_mode='stretch_both')
    tf_input     = TextInput(value="1",title="tf",sizing_mode='stretch_both')
    dt_input     = TextInput(value="0.01",title="dt",sizing_mode='stretch_both')
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
    
start_gui()
