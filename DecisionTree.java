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
*/
package decisiontree;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

public class DecisionTree 
{
    private TreeNode                                    m_root;
    private ArrayList<Features_record>                  m_arrDataSet;
    private Queue<TreeNode>                             m_Queue;
    private ArrayList<Features_record>                  m_data_train;
    private ArrayList<Features_record>                  m_data_test;
    private LinkedHashMap<String,ArrayList<String>>     m_mapPossibleValues;
    private int                                         m_nMaxDepth;
    
    
    public DecisionTree()
    {
        m_nMaxDepth = -1;
        m_root = new TreeNode(this,null);
        m_arrDataSet = new ArrayList<>();
        m_Queue = new LinkedList<TreeNode>();
        m_mapPossibleValues = new LinkedHashMap<String,ArrayList<String>>();
        m_data_train = new ArrayList<Features_record>();
        m_data_test = new ArrayList<Features_record>();
    }
    public void CreateDataHeader(String[] x_strFeatures)
    {
        for(String s: x_strFeatures)
            m_mapPossibleValues.put(s, new ArrayList<String>());
    }
    public void AddSample(String[] x_strSample,boolean x_bLabel,double weight)
    {
        if(x_strSample.length != m_mapPossibleValues.size())
        {
            System.out.printf("invalid feature vectore size(given %d,expected %d)!\r\n", x_strSample.length, m_mapPossibleValues.size());
            System.exit(-1);
        }
        
        String[] arrCols = getColumns();
        Features_record rec= new Features_record(arrCols);
        rec.SetLabel(x_bLabel);
        
        Iterator it = m_mapPossibleValues.keySet().iterator();
        for(int i = 0 ; i < x_strSample.length ; i++)
        {
            ArrayList<String> arrPossibleVals = (new ArrayList<ArrayList<String>>(m_mapPossibleValues.values())).get(i);
            String key = (String)it.next();
            if(!arrPossibleVals.contains(x_strSample[i]))
                arrPossibleVals.add(x_strSample[i]);
            rec.SetAttributeValue(key, x_strSample[i]);
        }
        rec.SetWeight(weight);
        m_data_train.add(rec);
    }
    public void AddSample(String[] x_strSample,boolean x_bLabel,boolean x_bTrainData)
    {
        if(x_strSample.length != m_mapPossibleValues.size())
        {
            System.out.printf("invalid feature vectore size(given %d,expected %d)!\r\n", x_strSample.length, m_mapPossibleValues.size());
            System.exit(-1);
        }
        
        String[] arrCols = getColumns();
        Features_record rec= new Features_record(arrCols);
        rec.SetLabel(x_bLabel);
        
        Iterator it = m_mapPossibleValues.keySet().iterator();
        for(int i = 0 ; i < x_strSample.length ; i++)
        {
            ArrayList<String> arrPossibleVals = (new ArrayList<ArrayList<String>>(m_mapPossibleValues.values())).get(i);
            String key = (String)it.next();
            if(!arrPossibleVals.contains(x_strSample[i]))
                arrPossibleVals.add(x_strSample[i]);
            rec.SetAttributeValue(key, x_strSample[i]);
        }
        if(x_bTrainData)
            m_data_train.add(rec);
        else 
            m_data_test.add(rec);
        
    }
    public double GetAccuracyOnTrainData()
    {
        double percentage = 0.0;
        int a =0;
        for(Features_record rec: m_data_train)
        {
            int nClass = (rec.GetLabel())? 1 : -1;
            String[] arrVals = rec.GetAttributeArray();
            if(Query(arrVals) != nClass)
                percentage += 1;
        }
        return (1.0 - (percentage / m_data_train.size())) * 100;
    }
    public double GetAccuracyOnTestData()
    {
        double percentage = 0.0;
        for(Features_record rec: m_data_test)
        {
            int nClass = (rec.GetLabel())? 1 : -1;
            String[] arrVals = rec.GetAttributeArray();
            if(Query(arrVals) != nClass)
                percentage += 1;
        }
        return (1.0 - (percentage / m_data_test.size())) * 100;
    }
    public TreeNode GetRoot()
    {
        return m_root;
    }
    private String[] getColumns()
    {
        String[] arrCols = new String[m_mapPossibleValues.size()];
        int i = 0;
        for (Map.Entry<String,ArrayList<String>> entry : m_mapPossibleValues.entrySet())
        {
            arrCols[i] = entry.getKey();
            i++;
        }
        return arrCols;
    }
    public void MakeTree()
    {
        //System.out.println("make tree");
        for(Features_record rec: m_data_train)
        {
            
            if(rec.GetLabel())
            {
                //System.out.printf("pos w =%f\r\n",rec.GetWeight());
                m_root.AddPositiveExample(rec);
            }
            else
            {
                //System.out.printf("neg w =%f\r\n",rec.GetWeight());
                m_root.AddNegativeExamples(rec);
            }
        }
        ArrayList<String> arrFeatures = new ArrayList<String>();
        String[] arrCols = getColumns();
        for(String s: arrCols)
            arrFeatures.add(s);
        m_root.SetExandingCandidates(arrFeatures);
        
        
        m_Queue.add(m_root);
        while(!m_Queue.isEmpty())
        {
            TreeNode node = m_Queue.remove();
            ProcessNode(node);            
        }        
    }
    public void MakeShallowTree(int x_nFeature)
    {
        for(Features_record rec: m_data_train)
        {            
            if(rec.GetLabel())
            {
                //System.out.printf("pos w =%f\r\n",rec.GetWeight());
                m_root.AddPositiveExample(rec);
            }
            else
            {
                //System.out.printf("neg w =%f\r\n",rec.GetWeight());
                m_root.AddNegativeExamples(rec);
            }
        }
        ArrayList<String> arrFeatures = new ArrayList<String>();
        String[] arrCols = getColumns();
        //for(String s: arrCols)
        //    arrFeatures.add(s);
        arrFeatures.add(arrCols[x_nFeature]);
        m_root.SetExandingCandidates(arrFeatures);
        
        
        m_Queue.add(m_root);
        while(!m_Queue.isEmpty())
        {
            TreeNode node = m_Queue.remove();
            ProcessNode(node);            
        }        
    }
    private void ProcessNode(TreeNode x_node)
    {
        if(x_node.IsLeaf())
        {
            //x_node.SetDecisionClass();
            x_node.SetDecisionClass_weighted();
            return;
        }
        if(m_nMaxDepth > 0 && GetNodeDepth(x_node) == m_nMaxDepth) //Don't split any furthur
        {
            // majority vote
            //x_node.SetDecisionClass();
            x_node.SetDecisionClass_weighted();
        }
        else if(x_node.GetCandidateCount() > 0)
        {
            //String strBestFeature = getBestAttribute(x_node);
            String strBestFeature = getBestAttribute_weighted(x_node);
            if(strBestFeature.equals("NOGAIN"))
            {
                x_node.SetNomoregain();
                x_node.SetDecisionClass_weighted();
                return;
            }
            ArrayList<TreeNode>arr_children = x_node.InsertChildrenBasedOnAttributeValues(strBestFeature);
            for(TreeNode n: arr_children)
                m_Queue.add(n);
        }
        else // majority vote
        {
            //x_node.SetDecisionClass();
            x_node.SetDecisionClass_weighted();
        }
    }
    private String getBestAttribute(TreeNode x_Node)
    {
        String[] arrCandidates = x_Node.GetExpandingCandidates();
        String strBestFeature = arrCandidates[0];
        double dBestInfo_gain = x_Node.GetInformationGain(strBestFeature);
        
        for(String s: arrCandidates)
        {
            if(x_Node.GetInformationGain(s) > dBestInfo_gain)
            {
                strBestFeature = s;
                dBestInfo_gain = x_Node.GetInformationGain(s);
            }
        }        
        return strBestFeature;
    }
    private String getBestAttribute_weighted(TreeNode x_Node)
    {
        String[] arrCandidates = x_Node.GetExpandingCandidates();
        String strBestFeature = arrCandidates[0];
        double dBestInfo_gain = x_Node.GetInformationGain_weighted(strBestFeature);
        
        for(String s: arrCandidates)
        {
            if(x_Node.GetInformationGain_weighted(s) > dBestInfo_gain)
            {
                strBestFeature = s;
                dBestInfo_gain = x_Node.GetInformationGain_weighted(s);
            }
        }
        if(dBestInfo_gain == 0)
            return "NOGAIN";
        return strBestFeature;
    }
    public String[] getAttributePossibleValues(String x_strAttrib)
    {
        if(m_mapPossibleValues.containsKey(x_strAttrib))
        {
            ArrayList<String> arr = (ArrayList<String>)m_mapPossibleValues.get(x_strAttrib);
            String[] arrPossibleVals = new String[arr.size()];
            arrPossibleVals = arr.toArray(arrPossibleVals);
            return arrPossibleVals;
        }
        else
        {
            System.out.printf("invalid attribute %s\r\n",x_strAttrib);
            System.exit(-1);
        }
        return null;
    }
    public int Query(String[] x_arrFeatures)
    {
        String[] arrCols = getColumns();
        Features_record rec= new Features_record(arrCols);
        Iterator it = m_mapPossibleValues.keySet().iterator();
        for(int i = 0 ; i < x_arrFeatures.length ; i++)
        {
            String key = (String)it.next();
            rec.SetAttributeValue(key, x_arrFeatures[i]);
        }
        if(m_root == null)
        {
            System.out.println("Corrupted Tree!");
            System.exit(-1);
        }
        TreeNode node = m_root;
        while(node != null)
        {
            if(node.IsLeaf())
                return node.getDecisionClass();
            
            if(IsMaximumdepthReached(node))
                return node.getDecisionClass();
            if(node.GetNomoreGain())
                return node.getDecisionClass();
            String strAttribute = node.GetAttribute();
            String strVal = rec.GetAttributeValue(strAttribute);
            node = node.getChild(strVal);
        }
        return -1;
    }
    public void PrintTree()
    {
        if(m_root != null)
            m_root.print();
    }
    public void SetMaxDepth(int x_nDepth)
    {
        m_nMaxDepth = x_nDepth;
    }
    public int GetNodeDepth(TreeNode x_Node)
    {
        if(x_Node == m_root)
            return 0;
        return 1 + GetNodeDepth(x_Node.getParent());
    }
    public boolean IsMaximumdepthReached(TreeNode x_Node)
    {
        if(m_nMaxDepth > 0)
        {
            if(GetNodeDepth(x_Node) == m_nMaxDepth)
                return true;
        }
        return false;
    }
    public void SetDataset_train(ArrayList<Features_record> x_arrDataSet)
    {
        for(Features_record rec : x_arrDataSet)
        {
            String[] arrAttribute = rec.GetAttributeArray();
            this.AddSample(arrAttribute, rec.GetLabel(), rec.GetWeight());
        }
    }
    public void SetDataset_test(ArrayList<Features_record> x_arrDataSet)
    {
        m_data_test = x_arrDataSet;
    }
    public void SetAllweightsEqual()
    {
        for(int i = 0 ; i < m_data_train.size() ;i++)
        {
            double dW = 1.0 / m_data_train.size();
            m_data_train.get(i).SetWeight(dW);
        }
    }
    public double GetWeighted_error()
    {
        double error = 0.0;
        for(Features_record rec: m_data_train)
        {
            int nClass = (rec.GetLabel())? 1 : -1;
            String[] arrVals = rec.GetAttributeArray();
            if(Query(arrVals) != nClass)
                error += rec.GetWeight();
        }
        return error;
    }
    public double GetAlpha()
    {
        double err = GetWeighted_error();
        return 0.5 * (Math.log((1-err)/err)/Math.log(Math.E));
    }
    private double getNormalization_factor()
    {
        double err = GetWeighted_error();
        return (2 * Math.sqrt(err * (1 - err)));
    }
    public ArrayList<Features_record> GetUpdatedDataset()
    {
        ArrayList<Features_record> arr_updated_dataset = new ArrayList<Features_record>();
        for(Features_record rec : m_data_train)
        {
            String[] strColumns = getColumns();
            Features_record r2 = new Features_record(strColumns);
            String[] arrAttributes = rec.GetAttributeArray();
            double alpha = GetAlpha();
            
            for(int i = 0 ; i < arrAttributes.length ; i++)
                r2.SetAttributeValue(strColumns[i],arrAttributes[i]);
            
            r2.SetLabel(rec.GetLabel());
            //--> Update AdaBoost Weight
            int nClass = (rec.GetLabel())? 1 : -1;
            double w_old = rec.GetWeight();
            double w_new = 0; 
            double Z = getNormalization_factor();
            if(Query(rec.GetAttributeArray()) == nClass)
                w_new = (w_old * Math.exp(-1 * alpha)) / Z;
            else
                w_new = (w_old * Math.exp(alpha)) / Z;
            //<-- Update AdaBoost Weight
            r2.SetWeight(w_new);
            //r2.SetWeight(1.0 / m_data_train.size());
            arr_updated_dataset.add(r2);
        }
        return arr_updated_dataset;
    }
    public double get_root_IG()
    {
        double dInfo_gain = m_root.GetCurrentEntropy_Weighted();
        for(TreeNode t: m_root.GetChildren())
        {
            double weight = t.GetTotalWeight() / m_root.GetTotalWeight();
            dInfo_gain -= weight * t.GetCurrentEntropy_Weighted();
        }
        
        return dInfo_gain;
    }
    public void MakeBinaryTree(int x_nFeature,boolean x_bLeftLabel,boolean x_bRightLabel)
    {
        ArrayList<String> arrFeatures = new ArrayList<String>();
        String[] arrCols = getColumns();

        arrFeatures.add(arrCols[x_nFeature]);
        m_root.SetExandingCandidates(arrFeatures);
        ArrayList<String> arrStr = new ArrayList<String>();
        arrStr.add("0");
        arrStr.add("1");
        m_mapPossibleValues.put(arrCols[x_nFeature],arrStr);
        m_root.AddPositiveExample(new Features_record(arrCols));
        m_root.AddNegativeExamples(new Features_record(arrCols));
        
        ArrayList<TreeNode>arr_children = m_root.InsertChildrenBasedOnAttributeValues(getColumns()[x_nFeature]);
        arr_children.get(0).ForceDecisionClass(x_bLeftLabel);
        arr_children.get(1).ForceDecisionClass(x_bRightLabel);
    }
    
}
