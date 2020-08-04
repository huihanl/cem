import matplotlib.pyplot as plt


def plot_history(history, env_id, epoch, expt_time):
    f, ax = plt.subplots(nrows=4, sharex=True, figsize=(20, 10))
    f.suptitle('{} at epoch {}, {:0.0f} seconds'.format(env_id, epoch, expt_time))
    ax[0].plot(history['epoch'], history['avg_ret'], label='population')
    ax[0].plot(history['epoch'], history['avg_ret_elites'], label='elite')
    ax[0].legend()
    ax[0].set_ylabel('average returns')

    ax[1].plot(history['epoch'], history['std_ret'], label='population')
    ax[1].plot(history['epoch'], history['std_ret_elites'], label='elites')
    ax[1].legend()
    ax[1].set_ylabel('standard deivation returns')

    ax[2].plot(history['epoch'], history['avg_suc'], label='population')
    ax[2].plot(history['epoch'], history['avg_suc_elites'], label='elite')
    ax[2].legend()
    ax[2].set_ylabel('average success rates')

    ax[3].plot(history['epoch'], history['std_suc'], label='population')
    ax[3].plot(history['epoch'], history['std_suc_elites'], label='elites')
    ax[3].legend()
    ax[3].set_ylabel('standard deivation success rates')

    ax[3].set_xlabel('epoch')

    f.savefig('./{}_small/learning_{}.png'.format(env_id, epoch))
    plt.clf()
