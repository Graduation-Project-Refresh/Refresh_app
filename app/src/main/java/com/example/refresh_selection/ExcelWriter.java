package com.example.refresh_selection;

import android.os.Build;
import android.util.Log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import androidx.annotation.RequiresApi;
import androidx.fragment.app.Fragment;


@RequiresApi(api = Build.VERSION_CODES.KITKAT)
public class ExcelWriter extends Fragment {
    BufferedWriter bw = null; // 출력 스트림 생성
    String NEWLINE =System.lineSeparator();
    String filePath = getContext().getFilesDir().getAbsolutePath();
//    ExcelWriter(){
//        File csv = new File("여기에 .csv파일의 절대경로를 입력한다");
//
//        try {
//            bw = new BufferedWriter(new FileWriter(csv, true));
//            bw.write("mlsfc,mcate_nm,Sex,Age,Month,time,day");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
    public void writeTitle(String name){
        File csv = new File(filePath+"\\"+name);
        BufferedWriter bw = null; // 출력 스트림 생성
        try {
            bw = new BufferedWriter(new FileWriter(csv, true));
            // csv파일의 기존 값에 이어쓰려면 위처럼 true를 지정하고, 기존 값을 덮어쓰려면 true를 삭제한다
            bw.write("mlsfc,mcate_nm,Sex,Age,Month,time,day");
            bw.write(NEWLINE);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) {
                    bw.flush(); // 남아있는 데이터까지 보내 준다
                    bw.close(); // 사용한 BufferedWriter를 닫아 준다
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    public void writeCSV(String row,String name) {
        // 입력된 내용 파일로 쓰기
        // Internal storage
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            Log.d("dataDirPath", "Internal private data dir: " + getContext().getDataDir().getAbsolutePath());
        }
        Log.d("fileDirPath", "Internal private file dir: " + getContext().getFilesDir().getAbsolutePath());
        Log.d("CacheDirPath", "Internal private cache dir: " + getContext().getCacheDir().getAbsolutePath());


        File csv = new File(filePath+"\\"+name);
        BufferedWriter bw = null; // 출력 스트림 생성
        try {
            bw = new BufferedWriter(new FileWriter(csv, true));
            // csv파일의 기존 값에 이어쓰려면 위처럼 true를 지정하고, 기존 값을 덮어쓰려면 true를 삭제한다
            bw.write("mlsfc,mcate_nm,Sex,Age,Month,time,day");
            bw.write(NEWLINE);
            bw.write(row);
            bw.write(NEWLINE);
            //getter setter로 가져오기
//            for (int i = 0; i < dataList.size(); i++) {
//                String[] data = dataList.get(i);
//                String aData = "";
//                aData = data[0] + "," + data[1] + "," + data[2] + "," + data[3];
//                // 한 줄에 넣을 각 데이터 사이에 ,를 넣는다
//                bw.write(aData);
//                // 작성한 데이터를 파일에 넣는다
//                bw.newLine(); // 개행
//            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) {
                    bw.flush(); // 남아있는 데이터까지 보내 준다
                    bw.close(); // 사용한 BufferedWriter를 닫아 준다
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    public List<List<String>> readCSV() {
        List<List<String>> csvList = new ArrayList<List<String>>();
        File csv = new File(filePath);
        BufferedReader br = null;
        String line = "";

        try {
            br = new BufferedReader(new FileReader(csv));
            while ((line = br.readLine()) != null) { // readLine()은 파일에서 개행된 한 줄의 데이터를 읽어온다.
                List<String> aLine = new ArrayList<String>();
                String[] lineArr = line.split(","); // 파일의 한 줄을 ,로 나누어 배열에 저장 후 리스트로 변환한다.
                aLine = Arrays.asList(lineArr);
                csvList.add(aLine);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close(); // 사용 후 BufferedReader를 닫아준다.
                }
            } catch(IOException e) {
                e.printStackTrace();
            }
        }
        return csvList;
    }

}
