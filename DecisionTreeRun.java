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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

public class DecisionTreeRun {

    public static void main(String[] args) 
    {
        //trainCoordinate_descent_heartDatset();
        trainAdaboost_heartDataset();
    }
    public static void trainCoordinate_descent_heartDatset()
    {
        String[] arrColumns = new String[22];
        for(int i = 0 ; i < 22 ; i++)
            arrColumns[i] = String.format("f%d",i);
        ArrayList<Features_record> dataset_train = new ArrayList<>();
        ArrayList<Features_record> dataset_test = new ArrayList<>();
        try
        {
            FileInputStream fstream = new FileInputStream("heart_train.data");
            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
            String strLine;
            int line = 1;
            while ((strLine = br.readLine()) != null)   
            {
                 String[] sample_vec = strLine.split(",");
                 if(sample_vec.length != 23)
                 {
                     System.out.printf("invalid record format on line %d\n",line);
                     System.exit(-1);
                 }
                 String[] feature_vec = new String[22];
                 for(int i = 1 ; i < 23 ; i++)
                     feature_vec[i-1] = sample_vec[i];
                 
                 boolean bLabel = (sample_vec[0].equals("1")) ? true : false;
                 Features_record rec = new Features_record(arrColumns, feature_vec);
                 rec.SetLabel(bLabel);
                 dataset_train.add(rec);
                 line++;
            }
            br.close();
            
            fstream = new FileInputStream("heart_test.data");
            br = new BufferedReader(new InputStreamReader(fstream));
            line = 1;
            while ((strLine = br.readLine()) != null)   
            {
                String[] sample_vec = strLine.split(",");
                if(sample_vec.length != 23)
                {
                    System.out.printf("invalid record format on line %d\n",line);
                    System.exit(-1);
                }
                String[] feature_vec = new String[22];
                for(int i = 1 ; i < 23 ; i++)
                    feature_vec[i-1] = sample_vec[i];
                
                boolean bLabel = (sample_vec[0].equals("1")) ? true : false;
                Features_record rec = new Features_record(arrColumns, feature_vec);
                rec.SetLabel(bLabel);
                dataset_test.add(rec);
                line++;
            }
            br.close();            
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
        }
      
        //CoordinateDescent codesc = new CoordinateDescent(200, arrColumns , dataset_train, dataset_train);
        //codesc.Run();
        //System.out.printf("M = %d, accuracy = %f, exp_loss = %f\r\n", codesc.getM(), codesc.get_accuracy(), codesc.calculate_loss());
        
        double old_loss = 0;
        double new_loss = 1;
        int M = 1;
        CoordinateDescent codesc = null;
        while (Math.abs(old_loss - new_loss) >= 0.01)
        {
            
            codesc = new CoordinateDescent(M, arrColumns , dataset_train, dataset_test);
            codesc.Run();
            System.out.printf("M = %d, accuracy_training = %f, accuracy_test = %f, exp_loss = %f\r\n", codesc.getM(), codesc.get_accuracy_train(),codesc.get_accuracy_test(), codesc.calculate_loss());

            old_loss = new_loss;
            new_loss = codesc.calculate_loss();
            M += 10;            
        } 
        codesc.printAlphas();
        
        
        
    }
    public static void trainAdaboost_heartDataset()
    {
        String[] arrColumns = new String[22];
        for(int i = 0 ; i < 22 ; i++)
            arrColumns[i] = String.format("f%d",i);
        ArrayList<Features_record> dataset_train = new ArrayList<>();
        ArrayList<Features_record> dataset_test = new ArrayList<>();
        try
        {
            FileInputStream fstream = new FileInputStream("heart_train.data");
            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
            String strLine;
            int line = 1;
            while ((strLine = br.readLine()) != null)   
            {
                 String[] sample_vec = strLine.split(",");
                 if(sample_vec.length != 23)
                 {
                     System.out.printf("invalid record format on line %d\n",line);
                     System.exit(-1);
                 }
                 String[] feature_vec = new String[22];
                 for(int i = 1 ; i < 23 ; i++)
                     feature_vec[i-1] = sample_vec[i];
                 
                 boolean bLabel = (sample_vec[0].equals("1")) ? true : false;
                 Features_record rec = new Features_record(arrColumns, feature_vec);
                 rec.SetLabel(bLabel);
                 dataset_train.add(rec);
                 line++;
            }
            br.close();
            
            fstream = new FileInputStream("heart_test.data");
            br = new BufferedReader(new InputStreamReader(fstream));
            line = 1;
            while ((strLine = br.readLine()) != null)   
            {
                String[] sample_vec = strLine.split(",");
                if(sample_vec.length != 23)
                {
                    System.out.printf("invalid record format on line %d\n",line);
                    System.exit(-1);
                }
                String[] feature_vec = new String[22];
                for(int i = 1 ; i < 23 ; i++)
                    feature_vec[i-1] = sample_vec[i];
                
                boolean bLabel = (sample_vec[0].equals("1")) ? true : false;
                Features_record rec = new Features_record(arrColumns, feature_vec);
                rec.SetLabel(bLabel);
                dataset_test.add(rec);
                line++;
            }
            br.close();            
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
        }
        AdaBooster booster = new AdaBooster(8);
        booster.SetColumns(arrColumns);
        booster.SetTrainingDataset(dataset_train);
        booster.SetTestDataset(dataset_test);
        booster.RunAdaboost();
        booster.printall();
        
        double accuracy = booster.getAccuracyOnTestData();
        System.out.println(accuracy);
        booster.print_errors();
        
        /*
        DecisionTree tree = new DecisionTree();
        tree.CreateDataHeader(arrColumns);

        tree.SetDataset_train(dataset_train);
        tree.SetDataset_test(dataset_test);
        tree.SetAllweightsEqual();
        
        tree.MakeTree();
        tree.PrintTree();
        double accuracy_train = tree.GetAccuracyOnTrainData();
        double accuracy_test = tree.GetAccuracyOnTestData();

        System.out.printf("training accuracy = %f, test_accuracy = %f\r\n", accuracy_train,accuracy_test);
*/
    }
    
    public static void testAdaboostOnTennis()
    {
        AdaBooster booster = new AdaBooster(10);
        booster.SetColumns(new String[]{"outlook", "temperature", "humidity", "wind"});
        Features_record[] arrRec = new Features_record[14];
        
        arrRec[0] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"sunny", "hot", "high", "FALSE"});
        arrRec[0].SetLabel(false);
        
        arrRec[1] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"sunny", "hot", "high", "TRUE"});
        arrRec[1].SetLabel(false);
        
        arrRec[2] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"overcast", "hot", "high", "FALSE"});
        arrRec[2].SetLabel(true);
        
        arrRec[3] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"rainy", "mild", "high", "FALSE" });
        arrRec[3].SetLabel(true);
        
        arrRec[4] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"rainy", "cool", "normal", "FALSE"});    
        arrRec[4].SetLabel(true);
        
        arrRec[5] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"rainy", "cool", "normal", "TRUE" });
        arrRec[5].SetLabel(false);
        
        arrRec[6] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"overcast", "cool", "normal", "TRUE"});
        arrRec[6].SetLabel(true);
        
        arrRec[7] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"sunny", "mild", "high", "FALSE"});
        arrRec[7].SetLabel(false);
        
        arrRec[8] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"sunny", "cool", "normal", "FALSE"});  
        arrRec[8].SetLabel(true);
        
        arrRec[9] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"rainy", "mild", "normal", "FALSE"});
        arrRec[9].SetLabel(true);
        
        arrRec[10] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"sunny", "mild", "normal", "TRUE"});  
        arrRec[10].SetLabel(true);
        
        arrRec[11] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"overcast", "mild", "high", "TRUE"});
        arrRec[11].SetLabel(true);
        
        arrRec[12] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"overcast", "hot", "normal", "FALSE"});    
        arrRec[12].SetLabel(true);
        
        arrRec[13] = new Features_record(new String[]{"outlook", "temperature", "humidity", "wind"},new String[]{"rainy", "mild", "high", "TRUE"});
        arrRec[13].SetLabel(false);
        
        ArrayList<Features_record> lstDataset = new ArrayList<Features_record>(Arrays.asList(arrRec));
        booster.SetTrainingDataset(lstDataset);
        booster.SetTestDataset(lstDataset);
        
        booster.RunAdaboost();
        
        booster.printall();
       
        double accuracy = booster.getAccuracyOnTestData();
        
        System.out.println(accuracy);
    }
    public static void testMushroom()
    {
        DecisionTree tree = new DecisionTree();
        tree.CreateDataHeader(new String[]{"cap_shape","cap_surface","cap_color",
                                           "bruises","odor","gill_attachment",
                                           "gill_spacing","gill_size","gill_color",
                                           "stalk_shape","stalk_root","stalk_surface_above_ring",
                                           "stalk_surface_below_ring","stalk_color_above_ring",
                                           "stalk_color_below_ring","veil_type","veil_color",
                                           "ring_number","ring_type","spore_print_color",
                                           "population","habitat"});
        try
        {
            FileInputStream fstream = new FileInputStream("mush_train.data");
            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
            String strLine;
            int line = 1;
            while ((strLine = br.readLine()) != null)   
            {
                 String[] sample_vec = strLine.split(",");
                 if(sample_vec.length != 23)
                 {
                     System.out.printf("invalid record format on line %d\n",line);
                     System.exit(-1);
                 }
                 String[] feature_vec = new String[22];
                 for(int i = 1 ; i < 23 ; i++)
                     feature_vec[i-1] = sample_vec[i];
                 if(!sample_vec[0].equals("p") && !sample_vec[0].equals("e"))
                 {
                     System.out.printf("invalid class %s on line %d\r\n", sample_vec[0],line);
                     System.exit(-1);
                 }
                 boolean bLabel = (sample_vec[0].equals("p")) ? true : false;
                 tree.AddSample(feature_vec, bLabel, true);
                 line++;
            }
            br.close();
            tree.MakeTree();
            tree.PrintTree();
            
            fstream = new FileInputStream("mush_test.data");
            br = new BufferedReader(new InputStreamReader(fstream));
            line = 1;
            while ((strLine = br.readLine()) != null)   
            {
                 String[] sample_vec = strLine.split(",");
                 if(sample_vec.length != 23)
                 {
                     System.out.printf("invalid record format on line %d\n",line);
                     System.exit(-1);
                 }
                 String[] feature_vec = new String[22];
                 for(int i = 1 ; i < 23 ; i++)
                     feature_vec[i-1] = sample_vec[i];
                 if(!sample_vec[0].equals("p") && !sample_vec[0].equals("e"))
                 {
                     System.out.printf("invalid class %s on line %d\r\n", sample_vec[0],line);
                     System.exit(-1);
                 }
                 boolean bLabel = (sample_vec[0].equals("p")) ? true : false;
                 tree.AddSample(feature_vec, bLabel, false);
                 line++;
            }
            br.close();
            double accuracy = tree.GetAccuracyOnTestData();
            System.out.printf("%f percent accuracy on test data\r\n",accuracy);
            
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
        }
    }
    public static void testTennis()
    {
        DecisionTree tree = new DecisionTree();
        tree.SetMaxDepth(1);
        
        tree.CreateDataHeader(new String[]{"outlook", "temperature", "humidity", "wind"});
        //Training data
        tree.AddSample(new String[]{"sunny", "hot", "high", "FALSE"}, false, true);
        tree.AddSample(new String[]{"sunny", "hot", "high", "TRUE"}, false, true);
        tree.AddSample(new String[]{"overcast", "hot", "high", "FALSE"},true, true);
        tree.AddSample(new String[]{"rainy", "mild", "high", "FALSE" },true, true);
        tree.AddSample(new String[]{"rainy", "cool", "normal", "FALSE"},true, true);    
        tree.AddSample(new String[]{"rainy", "cool", "normal", "TRUE" },false, true);
        tree.AddSample(new String[]{"overcast", "cool", "normal", "TRUE"},true, true);
        tree.AddSample(new String[]{"sunny", "mild", "high", "FALSE"},false, true);
        tree.AddSample(new String[]{"sunny", "cool", "normal", "FALSE"}, true, true);  
        tree.AddSample(new String[]{"rainy", "mild", "normal", "FALSE"},true, true);
        tree.AddSample(new String[]{"sunny", "mild", "normal", "TRUE"}, true, true);  
        tree.AddSample(new String[]{"overcast", "mild", "high", "TRUE"}, true, true);
        tree.AddSample(new String[]{"overcast", "hot", "normal", "FALSE"},true, true);    
        tree.AddSample(new String[]{"rainy", "mild", "high", "TRUE"},false, true);
        //Test data
        tree.AddSample(new String[]{"sunny", "hot", "high", "FALSE"}, false, false);
        tree.AddSample(new String[]{"sunny", "hot", "high", "TRUE"}, false, false);
        
        tree.AddSample(new String[]{"overcast", "hot", "high", "FALSE"},true, false);
        tree.AddSample(new String[]{"rainy", "mild", "high", "FALSE" },true, false);
        tree.AddSample(new String[]{"rainy", "cool", "normal", "FALSE"},true, false);    
        tree.AddSample(new String[]{"rainy", "cool", "normal", "TRUE" },false, false);
        tree.AddSample(new String[]{"overcast", "cool", "normal", "TRUE"},true, false);
        tree.AddSample(new String[]{"sunny", "mild", "high", "FALSE"},false, false);
        tree.AddSample(new String[]{"sunny", "cool", "normal", "FALSE"}, true, false);  
        tree.AddSample(new String[]{"rainy", "mild", "normal", "FALSE"},true, false);
        tree.AddSample(new String[]{"sunny", "mild", "normal", "TRUE"}, true, false);  
        tree.AddSample(new String[]{"overcast", "mild", "high", "TRUE"}, true, false);
        tree.AddSample(new String[]{"overcast", "hot", "normal", "FALSE"},true, false);    
        tree.AddSample(new String[]{"rainy", "mild", "high", "TRUE"},false, false);
        
        
        tree.SetAllweightsEqual();
        tree.MakeTree();
        tree.PrintTree();
        
        double accuracy = tree.GetAccuracyOnTestData();
        System.out.println(accuracy);
        
        DecisionTree tree2 = new DecisionTree();
        tree2.CreateDataHeader(new String[]{"outlook", "temperature", "humidity", "wind"});
        //tree2.SetMaxDepth(1);
        
        ArrayList<Features_record> ds2 = tree.GetUpdatedDataset();
        tree2.SetDataset_train(ds2);
        tree2.MakeTree();
        tree2.PrintTree();
        
        
        tree2.AddSample(new String[]{"sunny", "hot", "high", "FALSE"}, false, false);
        tree2.AddSample(new String[]{"sunny", "hot", "high", "TRUE"}, false, false);
        
        tree2.AddSample(new String[]{"overcast", "hot", "high", "FALSE"},true, false);
        tree2.AddSample(new String[]{"rainy", "mild", "high", "FALSE" },true, false);
        tree2.AddSample(new String[]{"rainy", "cool", "normal", "FALSE"},true, false);    
        tree2.AddSample(new String[]{"rainy", "cool", "normal", "TRUE" },false, false);
        tree2.AddSample(new String[]{"overcast", "cool", "normal", "TRUE"},true, false);
        tree2.AddSample(new String[]{"sunny", "mild", "high", "FALSE"},false, false);
        tree2.AddSample(new String[]{"sunny", "cool", "normal", "FALSE"}, true, false);  
        tree2.AddSample(new String[]{"rainy", "mild", "normal", "FALSE"},true, false);
        tree2.AddSample(new String[]{"sunny", "mild", "normal", "TRUE"}, true, false);  
        tree2.AddSample(new String[]{"overcast", "mild", "high", "TRUE"}, true, false);
        tree2.AddSample(new String[]{"overcast", "hot", "normal", "FALSE"},true, false);    
        tree2.AddSample(new String[]{"rainy", "mild", "high", "TRUE"},false, false);
        
                
        double accuracy2 = tree2.GetAccuracyOnTestData();
        System.out.println(accuracy2);
        
    }
}
