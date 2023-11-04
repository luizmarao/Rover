import csv
import os
import numpy as np
import pandas as pd

exps_root_folder = '/home/luiz/Experimentos_2023/4We'
goals = [-1, 1, 2]

contents = os.listdir(exps_root_folder)
exps_list = []
''' 
The restrictions bellow are a guidance to skip other experiments on the root folder. If empty, skips none.
As the experiments were planed with 5 seeds, and the names are follows a standard protocol, the following lines provide
useful options.
'''
exps_list_restrictions = []
# exps_list_restrictions = ['_001-', '_002-', '_003-', '_004-', '_005-']
# exps_list_restrictions = ['_006-', '_007-', '_008-', '_009-', '_010-']
# exps_list_restrictions = ['_011-', '_012-', '_013-', '_014-', '_015-']
# exps_list_restrictions = ['_016-', '_017-', '_018-', '_019-', '_020-']
# exps_list_restrictions = ['_021-', '_022-', '_023-', '_024-', '_025-']
# exps_list_restrictions = ['_026-', '_027-', '_028-', '_029-', '_030-']
# exps_list_restrictions = ['_031-', '_032-', '_033-', '_034-', '_035-']
# exps_list_restrictions = ['_036-', '_037-', '_038-', '_039-', '_040-']
# exps_list_restrictions = ['_041-', '_042-', '_043-', '_044-', '_045-']
# exps_list_restrictions = ['_046-', '_047-', '_048-', '_049-', '_050-']
# exps_list_restrictions = ['_051-', '_052-', '_053-', '_054-', '_055-']
# exps_list_restrictions = ['_056-', '_057-', '_058-', '_059-', '_060-']
# exps_list_restrictions = ['_061-', '_062-', '_063-', '_064-', '_065-']
# exps_list_restrictions = ['_066-', '_067-', '_068-', '_069-', '_070-']
# exps_list_restrictions = ['_071-', '_072-', '_073-', '_074-', '_075-']
# exps_list_restrictions = ['_076-', '_077-', '_078-', '_079-', '_080-']
# exps_list_restrictions = ['_081-', '_082-', '_083-', '_084-', '_085-']
# exps_list_restrictions = ['_086-', '_087-', '_088-', '_089-', '_090-']
# exps_list_restrictions = ['_091-', '_092-', '_093-', '_094-', '_095-']
# exps_list_restrictions = ['_096-', '_097-', '_098-', '_099-', '_100-']
# exps_list_restrictions = ['_101-', '_102-', '_103-', '_104-', '_105-']
# exps_list_restrictions = ['_106-', '_107-', '_108-', '_109-', '_110-']
# exps_list_restrictions = ['_111-', '_112-', '_113-', '_114-', '_115-']
# exps_list_restrictions = ['_116-', '_117-', '_118-', '_119-', '_120-']
# exps_list_restrictions = ['_121-', '_122-', '_123-', '_124-', '_125-']
# exps_list_restrictions = ['_126-', '_127-', '_128-', '_129-', '_130-']
# exps_list_restrictions = ['_131-', '_132-', '_133-', '_134-', '_135-']
# exps_list_restrictions = ['_136-', '_137-', '_138-', '_139-', '_140-']
# exps_list_restrictions = ['_141-', '_142-', '_143-', '_144-', '_145-']
# exps_list_restrictions = ['_146-', '_147-', '_148-', '_149-', '_150-']
# exps_list_restrictions = ['_151-', '_152-', '_153-', '_154-', '_155-']
# exps_list_restrictions = ['_156-', '_157-', '_158-', '_159-', '_160-']
# exps_list_restrictions = ['_161-', '_162-', '_163-', '_164-', '_165-']
# exps_list_restrictions = ['_166-', '_167-', '_168-', '_169-', '_170-']
# exps_list_restrictions = ['_171-', '_172-', '_173-', '_174-', '_175-']
# exps_list_restrictions = ['_176-', '_177-', '_178-', '_179-', '_180-']
# exps_list_restrictions = ['_181-', '_182-', '_183-', '_184-', '_185-']
# exps_list_restrictions = ['_186-', '_187-', '_188-', '_189-', '_190-']
# exps_list_restrictions = ['_191-', '_192-', '_193-', '_194-', '_195-']
# exps_list_restrictions = ['_196-', '_197-', '_198-', '_199-', '_200-']

eval_results = {}
for content in contents:
    content_path = os.path.join(exps_root_folder, content)
    if os.path.isdir(content_path):
        if len(exps_list_restrictions) != 0:
            for restriction in exps_list_restrictions:
                if restriction in content_path:
                    exps_list.append(content_path)
                    break
        else:
            exps_list.append(content_path)

for exp_path in exps_list:
    for goal in goals:
        if goal == -1:
            file_path = os.path.join(exp_path, 'Eval_complete_mode.csv')
        else:
            file_path = os.path.join(exp_path, 'Eval_goal{}_mode.csv'.format(goal))
        if goal not in eval_results.keys():
            eval_results[goal] = []
        if not os.path.exists(file_path):
            continue

        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                eval_results[goal].append(row)
                for key in reader.fieldnames:
                    if key == "Network":
                        eval_results[goal][-1][key] = exp_path.split('/')[-1][0:7] + "/" + eval_results[goal][-1][key]
                    else:
                        eval_results[goal][-1][key] = float(eval_results[goal][-1][key])


def adjust_column_width(writer, df, sheet_name):
    # for column in df:
    #     column_length = max(df[column].astype(str).map(len).max(), len(column))
    #     col_idx = df.columns.get_loc(column)
    #     writer.sheets[sheet_name].set_column(col_idx, col_idx, column_length+2)
    (max_row, max_col) = df.shape
    writer.sheets[sheet_name].autofilter(0, 0, max_row, max_col - 1)
    writer.sheets[sheet_name].autofit()


exps_full_track_df = pd.DataFrame(data=eval_results[-1])
exps_g1_df = pd.DataFrame(data=eval_results[1])
exps_g2_df = pd.DataFrame(data=eval_results[2])
exps_full_track_df = exps_full_track_df.sort_values('g2_reached_eps (%)', ascending=False)
exps_g1_df = exps_g1_df.sort_values('g1_reached_eps (%)', ascending=False)
exps_g2_df = exps_g2_df.sort_values('g2_reached_eps (%)', ascending=False)

writer = pd.ExcelWriter(exps_root_folder + '/compiled_results.xlsx', engine='xlsxwriter')
exps_full_track_df.to_excel(writer, sheet_name="Full Track", float_format="%.2f", index=False, freeze_panes=(1, 1))
exps_g1_df.to_excel(writer, sheet_name="Goal 1", float_format="%.2f", index=False, freeze_panes=(1, 1))
exps_g2_df.to_excel(writer, sheet_name="Goal 2", float_format="%.2f", index=False, freeze_panes=(1, 1))
adjust_column_width(writer, exps_full_track_df, "Full Track")
adjust_column_width(writer, exps_g1_df, "Goal 1")
adjust_column_width(writer, exps_g2_df, "Goal 2")
writer.save()

'''
compiled_results = dict(
    ep_len_full_means=[],
    ep_len_full_mins=[],
    ep_len_full_maxs=[],
    deaths_full_perc=[],
    deaths_g1_perc=[],
    deaths_g2_perc=[],
    g0r_perc=[],
    g1r_perc=[],
    g2r_perc=[],
    full_g1r_perc=[],
    full_g2r_perc=[],
)
for goal in eval_results.keys():
    for net in eval_results[goal]:
        if goal == -1:
            compiled_results['g0r_perc'].append(net['g0_reached_eps (%)'])
            compiled_results['full_g1r_perc'].append(net['g1_reached_eps (%)'])
            compiled_results['full_g2r_perc'].append(net['g2_reached_eps (%)'])
            compiled_results['deaths_full_perc'].append(net['deaths (%)'])
            compiled_results['ep_len_full_means'].append(net['ep_len_mean'])
            compiled_results['ep_len_full_mins'].append(net['ep_len_min'])
            compiled_results['ep_len_full_maxs'].append(net['ep_len_max'])
        else:
            compiled_results['g{}r_perc'.format(goal)].append(net['g{}_reached_eps (%)'.format(goal)])
            compiled_results['deaths_g{}_perc'.format(goal)].append(net['deaths (%)'])


deaths_full_perc_median = np.median(compiled_results['deaths_full_perc'])
deaths_g1_perc_median = np.median(compiled_results['deaths_g1_perc'])
deaths_g2_perc_median = np.median(compiled_results['deaths_g2_perc'])
g0r_perc_median = np.median(compiled_results['g0r_perc'])
g1r_perc_median = np.median(compiled_results['g1r_perc'])
g2r_perc_median = np.median(compiled_results['g2r_perc'])
full_g1r_perc_median = np.median(compiled_results['full_g1r_perc'])
full_g2r_perc_median = np.median(compiled_results['full_g2r_perc'])
deaths_full_perc_mean = np.mean(compiled_results['deaths_full_perc'])
deaths_g1_perc_mean = np.mean(compiled_results['deaths_g1_perc'])
deaths_g2_perc_mean = np.mean(compiled_results['deaths_g2_perc'])
g0r_perc_mean = np.mean(compiled_results['g0r_perc'])
g1r_perc_mean = np.mean(compiled_results['g1r_perc'])
g2r_perc_mean = np.mean(compiled_results['g2r_perc'])
full_g1r_perc_mean = np.mean(compiled_results['full_g1r_perc'])
full_g2r_perc_mean = np.mean(compiled_results['full_g2r_perc'])
deaths_full_perc_std = np.std(compiled_results['deaths_full_perc'])
deaths_g1_perc_std = np.std(compiled_results['deaths_g1_perc'])
deaths_g2_perc_std = np.std(compiled_results['deaths_g2_perc'])
g0r_perc_std = np.std(compiled_results['g0r_perc'])
g1r_perc_std = np.std(compiled_results['g1r_perc'])
g2r_perc_std = np.std(compiled_results['g2r_perc'])
full_g1r_perc_std = np.std(compiled_results['full_g1r_perc'])
full_g2r_perc_std = np.std(compiled_results['full_g2r_perc'])
deaths_full_perc_min = np.min(compiled_results['deaths_full_perc'])
deaths_g1_perc_min = np.min(compiled_results['deaths_g1_perc'])
deaths_g2_perc_min = np.min(compiled_results['deaths_g2_perc'])
g0r_perc_min = np.min(compiled_results['g0r_perc'])
g1r_perc_min = np.min(compiled_results['g1r_perc'])
g2r_perc_min = np.min(compiled_results['g2r_perc'])
full_g1r_perc_min = np.min(compiled_results['full_g1r_perc'])
full_g2r_perc_min = np.min(compiled_results['full_g2r_perc'])
deaths_full_perc_max = np.max(compiled_results['deaths_full_perc'])
deaths_g1_perc_max = np.max(compiled_results['deaths_g1_perc'])
deaths_g2_perc_max = np.max(compiled_results['deaths_g2_perc'])
g0r_perc_max = np.max(compiled_results['g0r_perc'])
g1r_perc_max = np.max(compiled_results['g1r_perc'])
g2r_perc_max = np.max(compiled_results['g2r_perc'])
full_g1r_perc_max = np.max(compiled_results['full_g1r_perc'])
full_g2r_perc_max = np.max(compiled_results['full_g2r_perc'])


ep_len_full_means_mean = np.mean(compiled_results['ep_len_full_means']) * 0.04
ep_len_full_means_median = np.median(compiled_results['ep_len_full_means']) * 0.04
ep_len_full_means_std = np.std(compiled_results['ep_len_full_means']) * 0.04
ep_len_full_mins_min = np.min(compiled_results['ep_len_full_mins']) * 0.04
ep_len_full_maxs_max = np.max(compiled_results['ep_len_full_maxs']) * 0.04

print("\\toprule")
print("\\textbf{Parameter}\t\t\t\t\t\t& \\textbf{Mean ± Std}\t& \\textbf{Median}\t& \\textbf{Max}\t& \\textbf{Min} \\\\")
print("\\midrule \\midrule")
print("Deaths in full track (\\%)\t\t\t\t&\t{:.3f} ± {:.3f}\t\t& {:.3f} \t\t\t& {:.3f} \t\t& {:.3f} \\\\ \\hline".format(deaths_full_perc_mean, deaths_full_perc_std, deaths_full_perc_median, deaths_full_perc_max, deaths_full_perc_min))
print("$goal_1$ reached in full track (\\%)\t\t&\t{:.3f} ± {:.3f}\t\t& {:.3f} \t\t\t& {:.3f} \t\t& {:.3f} \\\\ \\hline".format(full_g1r_perc_mean, full_g1r_perc_std, full_g1r_perc_median, full_g1r_perc_max, full_g1r_perc_min))
print("$goal_2$ reached in full track (\\%)\t\t&\t{:.3f} ± {:.3f}\t\t& {:.3f} \t\t\t& {:.3f} \t\t& {:.3f} \\\\ \\hline".format(full_g2r_perc_mean, full_g2r_perc_std, full_g2r_perc_median, full_g2r_perc_max, full_g2r_perc_min))
print("$goal_0$ reached individually (\\%)\t\t&\t{:.3f} ± {:.3f}\t\t& {:.3f} \t\t\t& {:.3f} \t\t& {:.3f} \\\\ \\hline".format(g0r_perc_mean, g0r_perc_std, g0r_perc_median, g0r_perc_max, g0r_perc_min))
print("$goal_1$ reached individually (\\%)\t\t&\t{:.3f} ± {:.3f}\t\t& {:.3f} \t\t\t& {:.3f} \t\t& {:.3f} \\\\ \\hline".format(g1r_perc_mean, g1r_perc_std, g1r_perc_median, g1r_perc_max, g1r_perc_min))
print("$goal_2$ reached individually (\\%)\t\t&\t{:.3f} ± {:.3f}\t\t& {:.3f} \t\t\t& {:.3f} \t\t& {:.3f} \\\\ \\hline".format(g2r_perc_mean, g2r_perc_std, g2r_perc_median, g2r_perc_max, g2r_perc_min))
print("Episode time in full track (\\%)\t\t\t&\t{:.3f} ± {:.3f}\t\t& {:.3f} \t\t\t& {:.3f} \t\t& {:.3f} \\\\".format(ep_len_full_means_mean, ep_len_full_means_std, ep_len_full_means_median, ep_len_full_maxs_max, ep_len_full_mins_min))
print("\\bottomrule")

fully_success_agents = 0
for nets in zip(compiled_results['full_g2r_perc'], compiled_results['g0r_perc'], compiled_results['g1r_perc'],
                compiled_results['g2r_perc']):
    if np.product(nets) == 100.0**4:
        fully_success_agents += 1

print()
print("\\toprule")
print("\\textbf{Parameter}\t\t\t\t\t\t& \\textbf{Mean ± Std}\t& \\textbf{Median}\t& \\textbf{Max}\t& \\textbf{Min} \\\\")
print("Episode time in full track (\\%)\t\t&\t{:.3f} ± {:.3f}\t\t& {:.3f} \t\t\t& {:.3f} \t\t& {:.3f} \\\\ \\hline".format(ep_len_full_means_mean, ep_len_full_means_std, ep_len_full_means_median, ep_len_full_maxs_max, ep_len_full_mins_min))
print("\\midrule \\midrule")
print("\\bottomrule")

'''
