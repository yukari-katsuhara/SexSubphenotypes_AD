import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import chi2_contingency
import scipy.stats as stats
from math import log10, log2


def ICD10_code_to_chapter(let):
    if let == 'nan':
        return 'NaN';
    elif let[0] == 'A' or let[0] == 'B':
        return 'A00–B99';
    elif let[0] == 'C' or (let[0] == 'D' and int(let[1])>=0 and int(let[1])<5):
        return 'C00–D48';
    elif let[0] == 'D' and int(let[1])>=5 and int(let[1])<9:
        return 'D50–D89';
    elif let[0] == 'E':
        return 'E00–E90';
    elif let[0] == 'H' and int(let[1])>=0 and int(let[1])<6:
        return 'H00–H59';
    elif let[0] == 'H' and int(let[1])>=6 and int(let[1])<=9:
        return 'H60–H95';
    elif let[0] == 'K':
        return 'K00–K93';
    elif let[0] == 'P':
        return 'P00–P96';
    elif let[0] == 'S' or let[0] == 'T':
        return 'S00–T98';
    elif let[0] in ['V','W','X','Y']:
        return 'V01–Y98';
    elif let[0] in ['F', 'G','I', 'J', 'L', 'M', 'N', 'O','Q','R','Z','U']:
        return '{}00–{}99'.format(let[0], let[0]);
    else:
        return let;
    
def ICDchapter_to_name(chp):
    if chp == 'nan': return 'NaN';
    elif chp == 'A00–B99': return 'Infectious';
    elif chp == 'C00–D48': return 'Neoplasms';
    elif chp == 'D50–D89': return 'Blood-Related Disorders';
    elif chp == 'E00–E90': return 'Endocrine, Nutritional and Metabolic Disorders';
    elif chp == 'F00–F99': return 'Mental and Behavioural Disorders';
    elif chp == 'G00–G99': return 'Diseases of Nervous System';
    elif chp == 'H00–H59': return 'Diseases of Eye and Adnexa';
    elif chp == 'H60–H95': return 'Diseases of Ear and Mastoid process';
    elif chp == 'I00–I99': return 'Diseases of Circulatory System';
    elif chp == 'J00–J99': return 'Diseases of Respiratory System';
    elif chp == 'K00–K93': return 'Diseases of Digestive System';
    elif chp == 'L00–L99': return 'Diseases of Skin and Subcutaneous Tissue';
    elif chp == 'M00–M99': return 'Musculoskeletal System Diseases';
    elif chp == 'N00–N99': return 'Genitourinary System Diseases';
    elif chp == 'O00–O99': return 'Pregnancy and Childbirth';
    elif chp == 'P00–P96': return 'Perinatal Diseases';
    elif chp == 'Q00–Q99': return 'Congenital Diseases';
    elif chp == 'R00–R99': return 'Abnormal Clinical and Lab Findings';
    elif chp == 'S00–T98': return 'Injury, Poisoning and External Issues';
    elif chp == 'V01–Y98': return 'External Causes';
    elif chp == 'Z00–Z99': return 'Health Status and Services';
    elif chp == 'U00–U99': return 'Codes for special purposes';
    else: return ' ';
    
def countPtsDiagnosis_Dict(df, total_pt):
    ptDiagCount = dict()
    for n in diagkeys:
        diagtemp = df[['PatientID',n]].drop_duplicates() # drop duplicate diagnosis for each patient
        ptDiagCount[n]= pd.DataFrame(diagtemp[n].value_counts()).reset_index()
        ptDiagCount[n].columns = [n,'Count']
        ptDiagCount[n]['Count_r'] = total_pt - ptDiagCount[n]['Count']
    return ptDiagCount

def sigTestCounts(allcounts, n = None, verbose = False, diag = False): 
    ''' 
    combined = sigTestCounts(allcounts, n = None, verbose = False, diag = False)
    Inputs:
        allcounts - dataframe or dictionary.
            Dataframe is of format index with feature name, and 4 columns 
            that make up the contingency table of interest in the order of
            case positive, case negative, control positive, control negative. 
            Each row will be reshaped into a 2x2 contingency table. 
            If dictionary of dataframes, will extract dataframe with key n
        n - default: None. 
            If allcounts is dictionary, n is the key to extract the dataframe
            of interest.
        verbose - default: False
        diag - default: False. If true, will append ICD10 category to the output
            dataframes.
    ----- 
    Outputs:
        combined - dataframe with fisher or chi square stats and odds ratios 
        appended. '''
    # First, for fischer choose any row with less than 5 patients in a category
    if type(allcounts) is dict:
        allcounts = allcounts[n]
    print(n, ': Amount: ', allcounts.shape[0])
    
    temp_less5 = allcounts[allcounts.min(axis=1) < 5]  # take all with counts less than 5
    fisher1 = pd.DataFrame()
    if temp_less5.shape[0] > 0:
        print('\t Fisher Exact for <5 pts in a category, num:', temp_less5.shape[0])
        fisher = temp_less5 \
            .apply(lambda x: stats.fisher_exact(np.array(x).reshape(2,2)), axis=1) \
            .apply(pd.Series)
        fisher.columns = ['OddsRatio', 'pvalue']
        if verbose: print('\t\t fisher:', fisher.shape)

        maxratio = fisher['OddsRatio'][fisher['OddsRatio'] < np.inf].max();
        minratio = fisher['OddsRatio'][fisher['OddsRatio'] > 0].min();
        fisher = fisher.replace(np.inf, maxratio + 1) 
        fisher['log2_oddsratio'] = fisher['OddsRatio'] \
            .apply(lambda x: log2(2**-11) if (x == 0) else log2(x))

        # Replace 0 p-value with a small non-zero value (min p-value / 2)
        minpvalue = fisher['pvalue'][fisher['pvalue'] > 0].min()
        fisher['pvalue'] = fisher['pvalue'].replace(0, minpvalue / 2)
        
        # Apply -log10 to p-values, making sure there are no negative or zero values
        fisher['-log_pvalue'] = fisher['pvalue'].apply(lambda x: -log10(x) if x > 0 else np.nan)
        
        fisher1 = fisher.merge(temp_less5, how='right', left_index=True, right_index=True)
        if verbose: print('\t\t fisher1:', fisher1.shape)

    # now take the rest of the patients
    temp_more5 = allcounts[allcounts.min(axis=1) >= 5]
    print('\t Chi square for >=5 pts in a category, num:', temp_more5.shape[0])

    fisher = temp_more5 \
        .apply(lambda x: stats.fisher_exact(np.array(x).reshape(2,2)), axis=1) \
        .apply(pd.Series)
    fisher.columns = ['OddsRatio', 'fpvalue']

    maxratio = fisher['OddsRatio'][fisher['OddsRatio'] < np.inf].max()
    minratio = fisher['OddsRatio'][fisher['OddsRatio'] > 0].min()
    fisher = fisher.replace(np.inf, maxratio + 1)
    fisher['log2_oddsratio'] = fisher['OddsRatio'] \
        .apply(lambda x: log2(2**-11) if (x == 0) else log2(x))
    
    # Replace 0 p-value with a small non-zero value
    fisher['fpvalue'] = fisher['fpvalue'].replace(0, np.nan)
    minpvalue = fisher['fpvalue'].min()
    fisher['fpvalue'].fillna(minpvalue / 2, inplace=True)
    
    # Apply -log10 to fp-values, ensuring no zero or negative values
    fisher['-log_fpvalue'] = fisher['fpvalue'].apply(lambda x: -np.log10(x) if x > 0 else np.nan)

    if verbose: print('\t\t fisher', fisher.shape)

    chisquare = temp_more5.apply(lambda x: chi2_contingency(np.array(x).reshape(2,2)), axis=1) \
                          .apply(pd.Series)
    chisquare.columns = ['chistat', 'pvalue', 'dof', 'expected']
    chisquare = chisquare.merge(temp_more5, how='right', left_index=True, right_index=True)
    
    # Replace 0 p-value with the smallest non-zero p-value divided by 2
    minpvalue = chisquare['pvalue'][chisquare['pvalue'] > 0].min()
    chisquare['pvalue'] = chisquare['pvalue'].replace(0, minpvalue / 2)
    chisquare['-log_pvalue'] = chisquare['pvalue'].apply(lambda x: -log10(x))
    if verbose: print('\t\t chisquare:', chisquare.shape)

    combined = chisquare.merge(fisher, left_index=True, right_index=True, how='left')
    combined = combined.append(fisher1)
    if verbose: print('\t\t combined 1:', combined.shape)
    
    if diag:
        temp = alldiag[[n, 'ICD10_chape']].drop_duplicates()  # create mapping
        temp = temp[temp['ICD10_chape'] != 'NaN'].groupby(n)['ICD10_chape'].apply(list)
        combined = combined.merge(temp, how='left', left_index=True, right_index=True, suffixes=(False, False))
        if verbose: print('\t\t combined 2:', combined.shape)

    print('\t Final num: ', combined.shape[0])
    
    return combined

def sigTestCountsDict(allcountsdict, dictkeys, verbose = False, diag = False):
    ''' combined = sigTestCountsDict(allcountsdict, verbose = False, diag = False)
    Inputs:
    allcountsdict - dictionary of dataframes.
        Each dataframe is of format index with feature name, and 4 columns 
        that make up the contingency table of interest in the order of
        case positive, case negative, control positive, control negative. 
        Each row will be reshaped into a 2x2 contingency table. 
    dictkeys - iterable object with the keys of allcountsdict.
    verbose - default: False
    diag - default: False. If true, will append ICD10 category to the output
        dataframes.
    ----
    Outputs: 
    combined - dictionary of dataframes with fisher or chi square 
    stats and odds ratios appended to each dataframe. '''
    
    combineddict = dict()
    for n in dictkeys:
        print('Significance testing on ', n)
        combineddict[n] = sigTestCounts(allcountsdict, n,verbose= verbose,diag = diag)
    return combineddict

def miami(df="dataframe", chromo=None, logp1=None, logp2=None, color=None, dim=(10,10), r=300, ar=90, gwas_sign_line=False,
          gwasp=0.05, dotsize=8, markeridcol=None, markernames=None, gfont=8, valpha=1, show=False, figtype='png',
          axxlabel=None, axylabel=None, axlabelfontsize=9, axlabelfontname="Arial", axtickfontsize=9, figtitle='miami plot',
          label1='firstgroup', label2='secondgroup', rand_colors=None,
          axtickfontname="Arial", ylm=None, gstyle=1, yskip=1, plotlabelrotation=0, figname='miami', invert=False, fig=None, ax=None):

    # Set axis labels
    _x, _y = 'Chromosomes', r'$ -log_{10}(P)$'

    # Compute -log10(P) values for the two groups
    df['tpval'] = df[logp1]  # First group's -log10(P) values
    df['tpval2'] = -df[logp2]  # Second group's -log10(P) values (negated for Miami plot)
    
    # Sort data by chromosome
    df = df.sort_values(chromo)
    
    # Assign an index for plotting
    df['ind'] = range(len(df))
    df_group = df.groupby(chromo)  # Group by chromosome for plotting

    # Set default color palette if not provided
    if rand_colors is None:
        rand_colors = ('#a7414a', '#282726', '#6a8a82', '#a37c27', '#563838', '#0584f2', '#f28a30', '#f05837',
                       '#6465a5', '#00743f', '#be9063', '#de8cf0', '#888c46', '#c0334d', '#270101', '#8d2f23',
                       '#ee6c81', '#65734b', '#14325c', '#704307', '#b5b3be', '#f67280', '#ffd082', '#ffd800',
                       '#ad62aa', '#21bf73', '#a0855b', '#5edfff', '#08ffc8', '#ca3e47', '#c9753d', '#6c5ce7')

    color_list = rand_colors[:df[chromo].nunique()]  # Assign colors to unique chromosomes

    xlabels = []  # Store chromosome labels for the x-axis
    xticks = []  # Store tick positions for the x-axis
    
    # Create a new figure if none is provided
    if fig is None:
        fig, (ax0, ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 20]}, figsize=dim)
        ax0.axis('off')  # Hide the left axis used for labeling groups
        fig.tight_layout()

    # Plot data points for each chromosome
    i = 0
    for label, df1 in df.groupby(chromo):
        df1.plot(kind='scatter', x='ind', y='tpval', color=color_list[i], s=dotsize, alpha=valpha, ax=ax)  # First group
        df1.plot(kind='scatter', x='ind', y='tpval2', color=color_list[i], s=dotsize, alpha=valpha, ax=ax)  # Second group
        
        # Determine midpoint for x-axis labeling
        df1_max_ind = df1['ind'].iloc[-1]
        df1_min_ind = df1['ind'].iloc[0]
        xlabels.append(label)  # Store chromosome label
        xticks.append((df1_max_ind - (df1_max_ind - df1_min_ind) / 2))  # Compute tick position
        
        i += 1  # Move to the next color in the palette

    # Add a horizontal line at y = 0 to separate the two groups
    ax.axhline(y=0, color='#7d7d7d', linewidth=.5, zorder=0)

    # Add GWAS significance threshold lines if specified
    if gwas_sign_line:
        ax.axhline(y=np.log10(gwasp), linestyle='--', color='#7d7d7d', linewidth=1)  # Upper threshold
        ax.axhline(y=-np.log10(gwasp), linestyle='--', color='#7d7d7d', linewidth=1)  # Lower threshold

    # Adjust margins to prevent unwanted whitespace
    ax.margins(x=0)
    ax.margins(y=0)
    
    # Set x-axis ticks
    ax.set_xticks(xticks)

    # Compute y-axis limits while ignoring NaN values
    ymin = np.nanmin(df['tpval2']) - 10
    ymax = np.nanmax(df['tpval']) + 10

    # Define y-axis tick intervals
    if ylm is not None:
        ylm = np.arange(ylm[0], ylm[1], ylm[2])  # Use specified range
    else:
        ylm = np.concatenate((np.arange(0, ymin, -yskip), np.arange(0, ymax, yskip)))  # Compute range automatically

    # Set y-axis limits
    ax.set_ylim([ymin, ymax])
    ax0.set_ylim([ymin, ymax])
    ax.set_yticks(ylm)  # Set y-axis tick positions
    
    # Add labels for each group on the left-side axis
    ax0.text(0, ymin/2, label2, fontsize=axlabelfontsize, fontname=axlabelfontname, rotation=90, va='center')
    ax0.text(0, ymax/2, label1, fontsize=axlabelfontsize, fontname=axlabelfontname, rotation=90, va='center')

    # Set x-axis tick labels
    ax.set_xticklabels(xlabels, rotation=ar)
    
    # Set y-axis tick labels
    ax.set_yticklabels(ylm.astype(int), fontsize=axtickfontsize, fontname=axtickfontname)

    # Update axis labels if specified
    if axxlabel:
        _x = axxlabel
    if axylabel:
        _y = axylabel
    ax.set_xlabel(_x, fontsize=axlabelfontsize, fontname=axlabelfontname)  # Set x-axis label
    
    ax.get_yaxis().get_label().set_visible(False)  # Hide y-axis label on the main plot
    
    # Add rotated y-axis label on the left-side axis
    ax0.text(.5, 0, _y, fontsize=axlabelfontsize, fontname=axlabelfontname, rotation=90, va='center')

    # Set plot title
    plt.title(figtitle)
    
    return fig, ax
