/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import java.util.ArrayList;

public class CoordinateDescent {
    String[]                    m_arrColumns;
    ArrayList<DecisionTree>     m_Trees;
    ArrayList<Features_record>  m_dataset_train;
    ArrayList<Features_record>  m_dataset_test;
    double[]                    m_Alphas;
    int                         M;
    public CoordinateDescent(int x_nM,String[] x_Columns,ArrayList<Features_record> x_dataset_train,ArrayList<Features_record>  x_dataset_test)
    {
        M = x_nM;
        m_Trees = new ArrayList<DecisionTree>();
        m_arrColumns = x_Columns;
        m_dataset_train = x_dataset_train;
        m_dataset_test = x_dataset_test;
        m_Alphas = new double[4*m_arrColumns.length];
        for(int i = 0 ; i < m_arrColumns.length ; i++)
        {
            DecisionTree tree00 = new DecisionTree();
            DecisionTree tree01 = new DecisionTree();
            DecisionTree tree10 = new DecisionTree();
            DecisionTree tree11 = new DecisionTree();
            tree00.CreateDataHeader(m_arrColumns);
            tree01.CreateDataHeader(m_arrColumns);
            tree10.CreateDataHeader(m_arrColumns);
            tree11.CreateDataHeader(m_arrColumns);
            
            tree00.MakeBinaryTree(i,false, false);
            tree01.MakeBinaryTree(i,false, true);
            tree10.MakeBinaryTree(i,true, false);
            tree11.MakeBinaryTree(i,true, true);
            m_Trees.add(tree00);
            m_Trees.add(tree01);
            m_Trees.add(tree10);
            m_Trees.add(tree11);
        } 
        for(int i = 0 ; i < 4*m_arrColumns.length ; i++)
            m_Alphas[i] = 0;
        
    }
    public void printTrees()
    {
        for(DecisionTree t : m_Trees)
        {
            t.PrintTree();
            System.out.printf("IG = %f\r\n", t.get_root_IG());
        }
    }
    public void Run()
    {
        for(int i = 0 ; i < M ; i++)
            Iterate();
    }
    private void Iterate()
    {
        for(int h = 0 ; h < m_Trees.size() ; h++)
        {
            double numerator = 0;
            double denominator = 0;
            for(int s = 0 ; s < m_dataset_train.size() ; s++)
            {
		int nPrediction = m_Trees.get(h).Query(m_dataset_train.get(s).GetAttributeArray());
                int nClass = (m_dataset_train.get(s).GetLabel()) ? 1 : -1;
		double sum = compute_sum_alphas(h,m_dataset_train.get(s).GetAttributeArray(),nClass);
		double e_to_sum = Math.exp(sum);
		if(nPrediction == nClass )
                {
                    numerator += e_to_sum;
                }
		else
                {
                    denominator += e_to_sum;
		}
            }            
            m_Alphas[h] = (0.5)*(Math.log(numerator/denominator)/Math.log(Math.E));
        }
    }
    private double compute_sum_alphas(int except_h, String[] x_arrVals,int x_nClass)
    {
        double sum = 0;
	for(int hh = 0 ; hh < m_Trees.size() ; hh++)
        {
            if(hh == except_h)
                continue;
            else if(m_Trees.get(hh).Query(x_arrVals) == x_nClass)
            {
                sum -= m_Alphas[hh];
            }
            else
            {
                sum += m_Alphas[hh];
            }	
        }
        return sum;
    }
    public double calculate_loss()
    {
	double loss = 0;
        for(int s = 0 ; s < m_dataset_train.size() ; s++)
	{
            double sum = 0;
            for(int h = 0 ; h < m_Trees.size() ; h++)
            {
                int nClass = (m_dataset_train.get(s).GetLabel()) ? 1 : -1;
		if(m_Trees.get(h).Query(m_dataset_train.get(s).GetAttributeArray()) == nClass)
                {
                    sum -= m_Alphas[h];
                }
		else
                {
                    sum += m_Alphas[h];
                }
            }
            loss += Math.exp(sum);
        }
		
	return loss;
    }
    public int classify(String[] x_arrFeatures)
    {   
        double sum = 0;
        for(int i = 0 ; i < m_Trees.size() ; i++)
            sum += (m_Trees.get(i).Query(x_arrFeatures)) * m_Alphas[i];
		return (short) (sum >= 0 ? 1 : -1);
    }
	
    public double get_accuracy_train()
    {
        double sum = 0.0;
        for(int i = 0 ; i < m_dataset_train.size() ; i++)
        {
            String[] arrVals = m_dataset_train.get(i).GetAttributeArray();
            int nClass = (m_dataset_train.get(i).GetLabel()) ? 1: -1;
            if(classify(arrVals) == nClass)
                sum += 1;
        }
        return 100 * (sum/m_dataset_train.size());
    }
    public double get_accuracy_test()
    {
        double sum = 0.0;
        for(int i = 0 ; i < m_dataset_test.size() ; i++)
        {
            String[] arrVals = m_dataset_test.get(i).GetAttributeArray();
            int nClass = (m_dataset_test.get(i).GetLabel()) ? 1: -1;
            if(classify(arrVals) == nClass)
                sum += 1;
        }
        return 100 * (sum/m_dataset_test.size());
    }
    public int getM()
    {
        return M;
    }
    public void printAlphas()
    {
        System.out.print("Alphas:\r\n");
        for(int i = 0 ; i < m_Alphas.length ;i++)
        {
            System.out.printf("%f\t", m_Alphas[i]);
            if(i % 5 == 4)
                System.out.println();
        }
        System.out.print("\r\n");
    }
}
