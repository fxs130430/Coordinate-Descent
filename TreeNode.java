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
import java.util.HashMap;
import java.util.Map;

public class TreeNode {
    private HashMap<String,TreeNode>    m_children;
    private ArrayList<String>           m_arrExpandingCandidates;  
    private ArrayList<Features_record>  m_arrPosExamples;
    private ArrayList<Features_record>  m_arrNegExamples;
    private String                      m_strAttribute;
    private int                         m_nClass;
    
    private DecisionTree                m_OwnerTree;
    private TreeNode                    m_Parent;
    private String                      m_strParentLink;
    private boolean                     m_bNomoreGain;
    
    public TreeNode(DecisionTree x_Owner,TreeNode x_Parent)
    {
        m_children = new HashMap();
        m_arrPosExamples = new ArrayList<>();
        m_arrNegExamples = new ArrayList<>();
        m_arrExpandingCandidates = new ArrayList<String>();
        m_strAttribute = "-";
        m_strParentLink = "-";
        m_nClass = 0;
        m_OwnerTree = x_Owner;
        m_Parent = x_Parent;
        m_bNomoreGain = false;
    }
    
    public ArrayList<TreeNode> InsertChildrenBasedOnAttributeValues(String x_strAttrib)
    {
        if(m_arrExpandingCandidates.contains(x_strAttrib))
        {
            m_arrExpandingCandidates.remove(x_strAttrib);
        }
        else
        {
            System.out.printf("Attribute %s doesn't exist in the list of candidates!\r\n",x_strAttrib);
            System.exit(-1);
        }
        m_strAttribute = x_strAttrib;
        String[] arrVals = m_OwnerTree.getAttributePossibleValues(x_strAttrib);
        ArrayList<TreeNode> arrRetNodes = new ArrayList<TreeNode>();
        for(int i = 0 ; i < arrVals.length ; i++)
        {
           TreeNode t = new TreeNode(m_OwnerTree,this);
           t.SetExandingCandidates(m_arrExpandingCandidates);
           m_children.put(arrVals[i], t);
           t.SetParentLink(arrVals[i]);
           arrRetNodes.add(t);
           for (Features_record rec : m_arrPosExamples) 
           {
               if(rec.GetAttributeValue(x_strAttrib).equals(arrVals[i]))
               {
                   if(rec.GetLabel())
                       t.AddPositiveExample(rec);
                   else
                       t.AddNegativeExamples(rec);
               }
           }
           for (Features_record rec : m_arrNegExamples) 
           {
               if(rec.GetAttributeValue(x_strAttrib).equals(arrVals[i]))
               {
                   if(rec.GetLabel())
                       t.AddPositiveExample(rec);
                   else
                       t.AddNegativeExamples(rec);
               }
           }
        } 
        return arrRetNodes;
    }
    public TreeNode getParent()
    {
        return m_Parent;
    }
    public TreeNode getChild(String x_strValue)
    {
        TreeNode node = null;
        ArrayList<TreeNode> arrChildren = GetChildren();
        for(TreeNode n: arrChildren)
        {
            if(n.GetParentLink().equals(x_strValue))
            {
                node = n;
                break;
            }
        }
        return node;
    }
    public ArrayList<TreeNode> GetChildren()
    {
        ArrayList<TreeNode> arrChildren = new ArrayList<TreeNode>();
        for (Map.Entry<String,TreeNode> entry : m_children.entrySet())
            arrChildren.add(entry.getValue());
        return arrChildren;        
    }
    public int GetCandidateCount()
    {
        return m_arrExpandingCandidates.size();
    }
    public String GetNodeAttribute()
    {
        return m_strAttribute;
    }
    public double GetInformationGain(String x_strFeature)
    {
        if(m_arrExpandingCandidates.contains(x_strFeature))
        {
            //m_arrExpandingCandidates.remove(x_strFeature);
        }
        else
        {
            System.out.printf("Attribute %s doesn't exist in the list of candidates!\r\n",x_strFeature);
            System.exit(-1);
        }
        String[] arrVals = m_OwnerTree.getAttributePossibleValues(x_strFeature);
        ArrayList<TreeNode> arrChildren_temp = new ArrayList<TreeNode>();
        for(int i = 0 ; i < arrVals.length ; i++)
        {
           TreeNode t = new TreeNode(m_OwnerTree,this);
           for (Features_record rec : m_arrPosExamples) 
           {
               if(rec.GetAttributeValue(x_strFeature).equals(arrVals[i]))
               {
                   if(rec.GetLabel())
                       t.AddPositiveExample(rec);
                   else
                       t.AddNegativeExamples(rec);
               }
           }
           for (Features_record rec : m_arrNegExamples) 
           {
               if(rec.GetAttributeValue(x_strFeature).equals(arrVals[i]))
               {
                   if(rec.GetLabel())
                       t.AddPositiveExample(rec);
                   else
                       t.AddNegativeExamples(rec);
               }
           }
           arrChildren_temp.add(t);
        }
        
        double dInfo_gain = GetCurrentEntropy();
        for(TreeNode t: arrChildren_temp)
        {
            double weight = t.GetTotalCount() / this.GetTotalCount();
            dInfo_gain -= weight * t.GetCurrentEntropy();
        }
        
        return dInfo_gain;
        
    }
    public double GetInformationGain_weighted(String x_strFeature)
    {
        if(m_arrExpandingCandidates.contains(x_strFeature))
        {
            //m_arrExpandingCandidates.remove(x_strFeature);
        }
        else
        {
            System.out.printf("Attribute %s doesn't exist in the list of candidates!\r\n",x_strFeature);
            System.exit(-1);
        }
        String[] arrVals = m_OwnerTree.getAttributePossibleValues(x_strFeature);
        ArrayList<TreeNode> arrChildren_temp = new ArrayList<TreeNode>();
        for(int i = 0 ; i < arrVals.length ; i++)
        {
           TreeNode t = new TreeNode(m_OwnerTree,this);
           for (Features_record rec : m_arrPosExamples) 
           {
               if(rec.GetAttributeValue(x_strFeature).equals(arrVals[i]))
               {
                   if(rec.GetLabel())
                       t.AddPositiveExample(rec);
                   else
                       t.AddNegativeExamples(rec);
               }
           }
           for (Features_record rec : m_arrNegExamples) 
           {
               if(rec.GetAttributeValue(x_strFeature).equals(arrVals[i]))
               {
                   if(rec.GetLabel())
                       t.AddPositiveExample(rec);
                   else
                       t.AddNegativeExamples(rec);
               }
           }
           arrChildren_temp.add(t);
        }
        
        double dInfo_gain = GetCurrentEntropy_Weighted();
        for(TreeNode t: arrChildren_temp)
        {
            double weight = t.GetTotalWeight() / this.GetTotalWeight();
            dInfo_gain -= weight * t.GetCurrentEntropy_Weighted();
        }
        
        return dInfo_gain;
        
    }
    public String GetParentLink()
    {
        return m_strParentLink;
    }
    public void SetParentLink(String x_strLink)
    {
        m_strParentLink = x_strLink;
    }
    public void SetDecisionClass()
    {
        //First two are perfect classification cases
        if(m_arrPosExamples.size() > 0 && m_arrNegExamples.size() == 0)
        {
            m_nClass = 1;
        }
        else if(m_arrPosExamples.size() == 0 && m_arrNegExamples.size() > 0)
        {
            m_nClass = -1;
        }        
        else if(m_arrPosExamples.size() > m_arrNegExamples.size())
        {
            m_nClass = 1;
        }
        else
        {
            m_nClass = -1;
        }
        
        
    }
    public void ForceDecisionClass(boolean x_bLabel)
    {
        m_nClass = (x_bLabel)? 1 : -1;
    }
    public void SetDecisionClass_weighted()
    {
        if(GetPositiveWeight() > GetNegativeWeight())
        {
            m_nClass = 1;
        }
        else
        {
            m_nClass = -1;
        }
    }
    public int getDecisionClass()
    {
        return m_nClass;
    }
    
    
    
    public String[] GetExpandingCandidates()
    {
        String[] arrCandidates = new String[m_arrExpandingCandidates.size()];
        for(int i = 0 ; i < m_arrExpandingCandidates.size() ; i++)
            arrCandidates[i] = m_arrExpandingCandidates.get(i);
        return arrCandidates;    
    }
    public void SetExandingCandidates(ArrayList<String> x_arrFeatures)
    {
        for(String s: x_arrFeatures)
            m_arrExpandingCandidates.add(s);
    }
    public boolean IsLeaf()
    {
        return (GetPositiveCount() == 0 || GetNegativeCount() == 0);
    }
    public void AddPositiveExample(Features_record x_Rec)
    {
        m_arrPosExamples.add(x_Rec);
    }
    public void AddNegativeExamples(Features_record x_Rec)
    {
        m_arrNegExamples.add(x_Rec);
    }
    public String GetAttribute()
    {
        return m_strAttribute;
    }
    public String GetParentAttribute()
    {
        if(m_Parent == null)
            return "-";
        return m_Parent.GetAttribute();
    }
    public boolean GetNomoreGain()
    {
        return m_bNomoreGain;
    }
    public void SetNomoregain()
    {
        m_bNomoreGain = true;
    }
    public double GetCurrentEntropy()
    {
        if(IsLeaf())
            return 0;
        double nPos = GetPositiveCount();
        double nNeg = GetNegativeCount();
        double p_pos = nPos / (nPos + nNeg);
        double p_neg = nNeg / (nPos + nNeg);
        return -(p_pos * log2(p_pos) + p_neg * log2(p_neg));
    }
    public double GetCurrentEntropy_Weighted()
    {
        if(IsLeaf())
            return 0;
        double dPos = GetPositiveWeight();
        double dNeg = GetNegativeWeight();
        double p_pos = dPos / (dPos + dNeg);
        double p_neg = dNeg / (dPos + dNeg);
        return -(p_pos * log2(p_pos) + p_neg * log2(p_neg));
    }
    public double GetPositiveCount()
    {
        return m_arrPosExamples.size();
    }
    public double GetNegativeCount()
    {
        return m_arrNegExamples.size();
    }
    public double GetPositiveWeight()
    {
        double dW= 0;
        for(Features_record rec : m_arrPosExamples)
            dW += rec.GetWeight();
        return dW;
    }
    public double GetNegativeWeight()
    {
        double dW= 0;
        for(Features_record rec : m_arrNegExamples)
            dW += rec.GetWeight();
        return dW;
    }
    public double GetTotalWeight()
    {
        return GetPositiveWeight() + GetNegativeWeight();
    }
    public double GetTotalCount()
    {
        return (GetPositiveCount() + GetNegativeCount());
    }
    private static double log2(double x_dNum)
    {
        return (Math.log10(x_dNum)/Math.log10(2));
    }
    public void print() {
        print("", true);
    }

    private void print(String prefix, boolean isTail) {
        if(IsLeaf())
            System.out.println(prefix + (isTail ? "└── " : "├── ") + m_strParentLink +","+" [pos("+(int)GetPositiveCount()+"), Neg("+(int)GetNegativeCount()+")]");
        else
            System.out.println(prefix + (isTail ? "└── " : "├── ") + m_strParentLink +","+m_strAttribute+" [pos("+(int)GetPositiveCount()+"), Neg("+(int)GetNegativeCount()+")]");            
        for (int i = 0; i < m_children.size() - 1; i++)
        {
            (new ArrayList<TreeNode>(m_children.values())).get(i).print(prefix + (isTail ? "    " : "│   "), false);
        }
        if (m_children.size() > 0) {
            (new ArrayList<TreeNode>(m_children.values())).get(m_children.size() - 1).print(prefix + (isTail ?"    " : "│   "), true);
        }
    }
}
