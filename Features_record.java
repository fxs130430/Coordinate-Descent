/*
%    Copyright 2016 by Farhad Shakerin
% 
%    Permission to use this software is granted subject to the 
%    following restrictions and understandings: 
% 
%    1. This material is for educational and research purposes only. 
% 
%    2. Farhad Shakerin has provided this software AS IS. Farhad
%       has made no warranty or representation that the 
%       operation of this software will be error-free, and he is 
%       under no obligation to provide any services, by way of 
%       maintenance, update, or otherwise. 
% 
%    3. Any user of such software agrees to indemnify and hold 
%       harmless Farhad Shakerin from 
%       all claims arising out of the use or misuse of this 
%       software, or arising out of any accident, injury, or damage 
%       whatsoever, and from all costs, counsel fees and liabilities 
%       incurred in or about any such claim, action, or proceeding 
%       brought thereon. 
% 
%    4. Users are requested, but not required, to inform Farhad Shakerin
%       of any noteworthy uses of this software.
*/package decisiontree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 *
 * @author asus
 */
public class Features_record {
    
    private LinkedHashMap   m_mapVals;
    private boolean         m_bLabel;
    private double          m_dWeight;
    public Features_record(String[] x_arrColumns)
    {
        m_mapVals = new LinkedHashMap();
        for(int i = 0 ; i < x_arrColumns.length ; i++)
            m_mapVals.put(x_arrColumns[i], "");
    }
    public Features_record(String[] x_arrKeys,String[] x_arrVals)
    {
        m_mapVals = new LinkedHashMap();
        for(int i = 0 ; i < x_arrKeys.length ; i++)
            m_mapVals.put(x_arrKeys[i], x_arrVals[i]);
    }
    
    public boolean GetLabel()
    {
        return m_bLabel;
    }
    public void SetLabel(boolean x_bLabel)
    {
        m_bLabel = x_bLabel;
    }
    public void SetWeight(double x_dWeight)
    {
        m_dWeight = x_dWeight;
    }
    public double GetWeight()
    {
        return m_dWeight;
    }
    public void SetAttributeValue(String x_strAttrib,String x_strVal)
    {
        if(m_mapVals.containsKey(x_strAttrib))
            m_mapVals.put(x_strAttrib, x_strVal);
        else
        {
            System.out.printf("invalid attribute %s\r\n",x_strAttrib);
            System.exit(-1);
        }
    }
    public String GetAttributeValue(String x_strAttrib)
    {
        String strVal = "";
        if(m_mapVals.containsKey(x_strAttrib))
        {
            strVal = (String)m_mapVals.get(x_strAttrib);
        }
        else
        {
            System.out.printf("invalid attribute %s\r\n",x_strAttrib);
            System.exit(-1);
        }
        return strVal;
    }  
    public String[] GetAttributeArray()
    {
        String[] arr = new String[m_mapVals.size()];
        for(int i = 0 ; i < m_mapVals.size() ; i++)
            arr[i] = (new ArrayList<String>(m_mapVals.values())).get(i);
        return arr;
    }
}
