from mpl_toolkits.axes_grid1 import make_axes_locatable
from seth_controller import SethController, EntityTypes, ENTITY_RADIUS
from pylab import *
import os

DPI = 90

def plot_state_history(savepath,controller,file_prefix) :
    robot = controller.trial_data['robot']
    
    ###########################
    figure()
    plot(robot.x_h,robot.y_h,'k,',lw=0.5,alpha=0.5,label=f'Robot Trajectory')
    plot(robot.x_h[-1],robot.y_h[-1],'k.',label=f'Robot Final Position',markersize=15.0)
    c = plt.Circle((robot.x_h[-1],robot.y_h[-1]), robot.RADIUS, color='k',fill=True)
    gca().add_patch(c)
    for entity_type in EntityTypes:
        colors = {
            EntityTypes.FOOD : 'g',
            EntityTypes.WATER : 'b',
            EntityTypes.TRAP : 'r',
        }
        for x,y in controller.trial_data[f'eaten_{entity_type.name}_positions'] :
            c = plt.Circle((x,y), ENTITY_RADIUS, color=colors[entity_type],fill=False)
            gca().add_patch(c)
        for x,y in controller.trial_data[f'uneaten_{entity_type.name}_positions'] :
            c = plt.Circle((x,y), ENTITY_RADIUS, color=colors[entity_type])
            gca().add_patch(c)
    xlim(-1.5,1.5)
    ylim(-1.5,1.5)
    gca().set_aspect('equal')
    savefig(os.path.join(savepath,f'{file_prefix}_spatial.png'),dpi=DPI)
    close()

    dur = max(controller.trial_data['sample_times'])
    
    ###########################
    figure()
    subplot2grid((4,1),(0,0))
    plot(controller.trial_data['sample_times'],controller.trial_data['water_battery_h'],'b-',label=f'water')
    plot(controller.trial_data['sample_times'],controller.trial_data['food_battery_h'],'g-',label=f'food')
    ylabel('batteries')
    ylim(0.0,3.5)
    xlim(0,dur)
    xticks([])
    legend()

    ###########################
    subplot2grid((4,1),(1,0))
    plot(controller.trial_data['sample_times'],controller.trial_data['score_h'],'k-',label=f'score')
    ylabel('score')
    # ylim(0.0,3.5)
    xlim(0,dur)
    xticks([])
    legend()
    
    ###########################
    subplot2grid((4,1),(2,0))
    for entity_type in EntityTypes:
        colors = {
            EntityTypes.FOOD : '#00ff00',
            EntityTypes.WATER : '#0000ff',
            EntityTypes.TRAP : '#ff0000',
        }
        s_h = np.array(robot.sensors_h[entity_type])
        plot(controller.trial_data['sample_times'],s_h[:,0],color=colors[entity_type],ls='-')
        plot(controller.trial_data['sample_times'],s_h[:,1],color=colors[entity_type],ls='--')
    xticks([])
    ylim(0,1)
    xlim(0,dur)
    ylabel('sensors')

    subplot2grid((4,1),(3,0))
    plot(controller.trial_data['sample_times'],robot.lm_h,color='k',ls='-',label='left')
    plot(controller.trial_data['sample_times'],robot.rm_h,color='k',ls='--',label='right')
    ylim(-3,3)
    xlim(0,dur)
    ylabel('motors')
    savefig(os.path.join(savepath,f'{file_prefix}_timeseries.png'),dpi=DPI)
    tight_layout()
    close()

def fitness_plots(savepath,pop_fit_history) :
    figure() ## detailed fitness plot
    n_generations = np.shape(pop_fit_history)[0]
    pop_size = np.shape(pop_fit_history)[1]
    im = imshow(np.array(pop_fit_history).T,aspect='auto',extent=[0,n_generations,0,pop_size])
    xticks(range(n_generations))    
    xlabel('generation')
    ylabel('individual (sorted by fitness)')
    title('fitness')
    ## draw colorbar
    divider = make_axes_locatable(gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    tight_layout()
    savefig(os.path.join(savepath,'fitness_history.png'),dpi=DPI)
    close()

    figure() ## summary of fitness plot (mean and max)
    means = np.mean(np.array(pop_fit_history),axis=1)
    stddevs = np.std(np.array(pop_fit_history),axis=1)
    maxes = np.max(np.array(pop_fit_history),axis=1)
    #plot(means,'k',label='mean fitness')
    errorbar(range(n_generations),means,stddevs,label='mean and std fitness ')
    plot(maxes,'r',label='max fitness')#,alpha=0.5,lw=0.5)
    if(n_generations > 1) :
        xlim(0,n_generations-1)
    legend()
    xlabel('generation');ylabel('fitness')
    tight_layout()
    savefig(os.path.join(savepath,'fitness_history_summary.png'),dpi=DPI)
    close()


def plot_population_genepool(savepath,pop) :
        figure() # genes_of_entire_population
        pop_size = len(pop)
        g = np.zeros((pop[0].N_GENES,pop_size))
        for index in range(pop_size) :
            g[:,index] = pop[index].genome
        imshow(g)
        xlabel('individual')
        ylabel('gene')
        colorbar()
        tight_layout()
        savefig(os.path.join(savepath,'population_genepool.png'),dpi=DPI)
        close()
