package com.example.refresh_selection;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.MenuItem;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.HurlStack;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.InputStreamVolleyRequest;
import com.google.android.material.bottomnavigation.BottomNavigationView;

import android.os.AsyncTask;

import android.os.Build;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;


import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
//
//import io.grpc.ManagedChannel;
//import io.grpc.ManagedChannelBuilder;

import java.io.File;

import static android.os.Environment.DIRECTORY_DOWNLOADS;


public class MainActivity_travel extends AppCompatActivity implements NavigationHost {

    private static final int GPS_ENABLE_REQUEST_CODE = 2001;
    private static final int PERMISSIONS_REQUEST_CODE = 100;
    String[] REQUIRED_PERMISSIONS = {
            Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private BottomNavigationView bottomNavigationView; // 바텀네비게이션 뷰
    private FragmentManager manager;
    private FragmentTransaction transaction;

    private FragmentMain fragmentMain;//프레임 레이아웃을 액티미티 메인 xml
    private FragmentMap fragmentMap;//프레임 레이아웃을 액티비티 맵 xml
    private FragmentProfile fragmentProfile;//프레임 레이아웃을 액티비티 프로파일 xml

    private DeepFM deepFM;
    Handler handler = null;
    private String acc_f1 = "";
    private String mUsername;
    private int model_version;
    private int currentRound = 0;
    private int max_train_round = 0;
    private Response response = null;
    private String responseString;
    private String return_task;
    private JSONObject response_json = null;
    private String address = "http://18.218.2.153:8089/api/";

    private File mydir;

    InputStreamVolleyRequest request;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_trip_main);

        //GpsTracker 위치활용 동의
        if (checkLocationServicesStatus()) {
            checkRunTimePermission();
        } else {
            showDialogForLocationServiceSetting();
        }

//        String Output_file_path = "/storage/self/primary/Download/save_model";
//        File check_file = new File(Output_file_path);
//
//        if(!check_file.exists()) {
//            boolean success = check_file.mkdir();
//        }
//
//        int count;

        try {

            oninit();

            Log.d("INFO", "success init");

            // download model (server -> client)
            download();

            // first model build
            deepFM.buildModel("/storage/self/primary/Download/save_model");
//            acc_f1 = deepFM.eval();
            trainOneRound(currentRound, "tt", 1);

//            AssetManager as = getResources().getAssets();
//            InputStream is = as.open("MyMultiLayerNetwork_beta6.zip");
//
//            OutputStream output = new FileOutputStream(check_file+"/MyMultiLayerNetwork_beta6.zip");
//            byte data[] = new byte[10240];
//
////                long total = 0;
//
//            while ((count = is.read(data)) != -1) {
//                // writing data to file
//                output.write(data, 0, count);
//            }
//
//            // flushing output
//            output.flush();
//
//            // closing streams
//            output.close();
//            is.close();
        }catch (Exception e) {
            Log.e("Error: ", e.getMessage());
        }

//        deepFM = new DeepFM(); // 수정해야됨 https://github.com/HwangDongJun/FederatedLearning-mobile_client/blob/02b0860549c0a88f34309cb64edac244f487cc28/test_deeplearning4j/app/src/main/java/com/example/test_deeplearning4j/MainActivity.java#L403 참고
//        deepFM.buildModel(Output_file_path);
//        acc_f1 = deepFM.eval();

        bottomNavigationView = (BottomNavigationView) findViewById(R.id.trip_navigationView);//바텀 네비게이션
        //바텀네비게이션 선택한 아이템이 있을때
        bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
                switch (menuItem.getItemId()) {

                    case R.id.HomeItem:
                        setFrag(0);
                        break;
                    case R.id.MapItem:
                        setFrag(1);
                        break;
                    case R.id.MyPageItem:
                        setFrag(2);
                        break;
                }
                return true;
            }
        });

        fragmentMain = new FragmentMain();
        fragmentMap = new FragmentMap();
        fragmentProfile = new FragmentProfile();

        setFrag(0); // 첫화면 설정

    }

    // 프래그먼트 교체가 일어나는 메서드
    private void setFrag(int n) {

        manager = getSupportFragmentManager();
        transaction = manager.beginTransaction();

        switch (n) {
            case 0:

                transaction.replace(R.id.frameLayout, fragmentMain);
                transaction.commit();
                break;
            case 1:

                transaction.replace(R.id.frameLayout, fragmentMap);
                transaction.commit();
                break;
            case 2:
                transaction.replace(R.id.frameLayout, fragmentProfile);
                transaction.commit();
                break;
        }
    }


    /**
     * Navigate to the given fragment.
     *
     * @param fragment       Fragment to navigate to.
     * @param addToBackstack Whether or not the current fragment should be added to the backstack.
     */
    @Override
    public void navigateTo(Fragment fragment, boolean addToBackstack) {
        FragmentTransaction transaction =
                getSupportFragmentManager()
                        .beginTransaction()
                        .replace(R.id.container, fragment);

        if (addToBackstack) {
            transaction.addToBackStack(null);
        }

        transaction.commit();
    }

    private void showDialogForLocationServiceSetting() {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity_travel.this);
        builder.setTitle("위치 서비스 비활성화");
        builder.setMessage("앱을 사용하기 위해서는 위치 서비스가 필요합니다.\n" + "위치 설정을 수정하실래요?");
        builder.setCancelable(true);
        builder.setPositiveButton("설정", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int id) {
                Intent callGPSSettingIntent = new Intent(android.provider.Settings.ACTION_LOCATION_SOURCE_SETTINGS);
                startActivityForResult(callGPSSettingIntent, GPS_ENABLE_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("취소", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int id) {
                dialog.cancel();
            }
        });
        builder.create().show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case GPS_ENABLE_REQUEST_CODE: //사용자가 GPS 활성 시켰는지 검사
                if (checkLocationServicesStatus()) {
                    if (checkLocationServicesStatus()) {
                        Log.d("@@@", "onActivityResult : GPS 활성화 되있음");
                        checkRunTimePermission();
                        return;
                    }
                }
                break;
        }
    }

    void checkRunTimePermission() {

        //런타임 퍼미션 처리
        // 1. 위치 퍼미션을 가지고 있는지 체크합니다.
        int hasFineLocationPermission = ContextCompat.checkSelfPermission(MainActivity_travel.this,
                Manifest.permission.ACCESS_FINE_LOCATION);
        int hasCoarseLocationPermission = ContextCompat.checkSelfPermission(MainActivity_travel.this,
                Manifest.permission.ACCESS_COARSE_LOCATION);

        // 내장/외장 메모리 퍼미션
        int hasReadPermission = ContextCompat.checkSelfPermission(MainActivity_travel.this,
                Manifest.permission.READ_EXTERNAL_STORAGE);
        int hasWritePermission = ContextCompat.checkSelfPermission(MainActivity_travel.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE);


        if (hasFineLocationPermission == PackageManager.PERMISSION_GRANTED &&
                hasCoarseLocationPermission == PackageManager.PERMISSION_GRANTED &&
                hasReadPermission == PackageManager.PERMISSION_GRANTED &&
                hasWritePermission == PackageManager.PERMISSION_GRANTED) {

            // 2. 이미 퍼미션을 가지고 있다면
            // ( 안드로이드 6.0 이하 버전은 런타임 퍼미션이 필요없기 때문에 이미 허용된 걸로 인식합니다.)


            // 3.  위치 값을 가져올 수 있음


        } else {  //2. 퍼미션 요청을 허용한 적이 없다면 퍼미션 요청이 필요합니다. 2가지 경우(3-1, 4-1)가 있습니다.

            // 3-1. 사용자가 퍼미션 거부를 한 적이 있는 경우에는
            if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity_travel.this, REQUIRED_PERMISSIONS[0])) {

                // 3-2. 요청을 진행하기 전에 사용자가에게 퍼미션이 필요한 이유를 설명해줄 필요가 있습니다.
                Toast.makeText(MainActivity_travel.this, "이 앱을 실행하려면 위치 접근 권한이 필요합니다.", Toast.LENGTH_LONG).show();
                // 3-3. 사용자게에 퍼미션 요청을 합니다. 요청 결과는 onRequestPermissionResult에서 수신됩니다.
                ActivityCompat.requestPermissions(MainActivity_travel.this, REQUIRED_PERMISSIONS,
                        PERMISSIONS_REQUEST_CODE);


            } else {
                // 4-1. 사용자가 퍼미션 거부를 한 적이 없는 경우에는 퍼미션 요청을 바로 합니다.
                // 요청 결과는 onRequestPermissionResult에서 수신됩니다.
                ActivityCompat.requestPermissions(MainActivity_travel.this, REQUIRED_PERMISSIONS,
                        PERMISSIONS_REQUEST_CODE);
            }

        }

    }

    public boolean checkLocationServicesStatus() {
        LocationManager locationManager = (LocationManager) getSystemService(LOCATION_SERVICE);

        return locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
                || locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER);
    }


    private void oninit() throws ExecutionException, InterruptedException {
        AsyncTask<Void, Integer, String> init_task = new OninitTask().execute();
        return_task = init_task.get();
        init_task.cancel(true);
        if(return_task == "NOT REGISTER CLIENT") {
            Log.d("INFO", "Not register client");

            // 예외상황 처리해야함
        }
    }

//    private void update() throws ExecutionException, InterruptedException {
//        AsyncTask<Void, Integer, String> update_task = new RequestUpdateTask().execute();
//        return_task = update_task.get();
//        update_task.cancel(true);
//    }
//
    private void download() throws ExecutionException, InterruptedException {
        AsyncTask<Void, String, String> download_task = new DownloadFileFromURL().execute();
        return_task = download_task.get();
        download_task.cancel(true);
    }
//
//    private void version() throws ExecutionException, InterruptedException {
//        AsyncTask<Void, Integer, String> version_task = new CheckModelVersion().execute();
//        return_task = version_task.get();
//        version_task.cancel(true);
//    }
//


    class DownloadFileFromURL extends AsyncTask<Void, String, String> implements Response.Listener<byte[]>, Response.ErrorListener {
        @Override
        protected String doInBackground(Void... voids) {
            RequestQueue queue = Volley.newRequestQueue(MainActivity_travel.this);
            String mUrl= address+"getModel";
            InputStreamVolleyRequest request = new InputStreamVolleyRequest(Request.Method.GET, mUrl,
                    new Response.Listener<byte[]>() {
                        @Override
                        public void onResponse(byte[] response) {
                            // TODO handle the response
                            try {
                                if (response!=null) {
                                    FileOutputStream outputStream;
                                    String name="get_model.zip";
                                    outputStream = openFileOutput(name, Context.MODE_PRIVATE);
                                    outputStream.write(response);
                                    outputStream.close();
//                                    Toast.makeText(this, "Download complete.", Toast.LENGTH_LONG).show();
                                }
                            } catch (Exception e) {
                                // TODO Auto-generated catch block
                                Log.d("KEY_ERROR", "UNABLE TO DOWNLOAD FILE");
                                e.printStackTrace();
                            }
                        }
                    } ,new Response.ErrorListener() {

                @Override
                public void onErrorResponse(VolleyError error) {
                    error.printStackTrace();
                }
            }, null);
            RequestQueue mRequestQueue = Volley.newRequestQueue(getApplicationContext(), new HurlStack());
            mRequestQueue.add(request);

//            String url = address+"getModel"; //서버url
//            StringRequest stringRequest = new StringRequest(
//                    Request.Method.GET,
//                    url, new Response.Listener<String>() {
//                @Override
//                public void onResponse(byte[] response) {
//                    // 모델 받아와서 저장하는 거 있어야 됨
//                    try {
//                        System.out.print("response --> : "+response);
//                        JSONObject jsonResponse = new JSONObject(response.toString());
////                        int count;
////                        try {
////
////                            AssetManager as = getResources().getAssets();
////                            InputStream is = as.open("MyMultiLayerNetwork_beta6.zip");
////
////                            OutputStream output = new FileOutputStream("/storage/self/primary/Download/save_model/MyMultiLayerNetwork_beta6.zip");
////                            byte data[] = new byte[10240];
////
////                            while ((count = is.read(data)) != -1) {
////                                // writing data to file
////                                output.write(data, 0, count);
////                            }
////
////                            // flushing output
////                            output.flush();
////
////                            // closing streams
////                            output.close();
////                            is.close();
////                        }catch (Exception e) {
////                            Log.e("Error: ", e.getMessage());
////                        }
//
//                    } catch (JSONException e) {
//                        e.printStackTrace();
//                    }
//                }
//            }, new Response.ErrorListener() {
//                @Override
//                public void onErrorResponse(VolleyError error) {
//                    error.printStackTrace();
//                }
//            });
//
//            queue.add(stringRequest);

            return "finish download";
        }

        @Override
        public void onErrorResponse(VolleyError error) {

        }

        @Override
        public void onResponse(byte[] response) {

        }
    }
//
//    class WakeupClientTask extends AsyncTask<String, Integer, String> {
//
//
//        @Override
//        protected String doInBackground(String... params) {
//            String host = params[0];
//            int port = Integer.parseInt(params[1]);
//
//            String bsic = BatteryStatus_IsCharging();
//            String bat_pct = bsic.split(",")[0];
//            String charging = bsic.split(",")[1];
//            Boolean wc = WifiConnected();
//            String csds = ClassSize_DataSize();
//
//
//            Request request = new Request.Builder()
//                    .url("http://" + address + ":8891/client_wake_up?client_name=" + mUsername + "&battery_pct=" + bat_pct + "&is_charging=" + charging
//                            + "&wifi_conn=" + Boolean.toString(wc) + "&classsize_datasize=" + csds)
//                    .build();
//
//            try {
//                response = client.newCall(request).execute();
//                if (response.isSuccessful()) {
//                    ResponseBody body = response.body();
//                    if (body != null) {
//                        responseString = body.string();
//                    }
//                }
//                else
//                    Log.d("INFO", "Connect Error Occurred");
//            } catch (IOException e) {
//                e.printStackTrace();
//            } finally {
//                response.body().close();
//            }
//
//            return responseString;
//        }
//
//        @Override
//        protected void onPostExecute(String result) {
////            logArea.append("send wake up \n");
//            Log.d("INFO", "send wake up");
//            Log.d("INFO", "Response from the server : " + responseString);
//        }
//    }
//
//
    class OninitTask extends AsyncTask<Void, Integer, String> {
        @Override
        protected String doInBackground(Void... voids) {
            // client ready!
            String ModelName = Build.MODEL;

            String Output_file_path = "/storage/self/primary/Download/save_model";
            File check_file = new File(Output_file_path);
            String data_path = "/storage/self/primary/Download/data_balance/client1_train/";
            File check_file_file = new File(data_path);
            // 맨 처음에 서버에서 모델 받아오는거. 서버에서 모델 받아와서 Output_file_path에 저장함
            if(!check_file.exists()) {
                boolean success = check_file.mkdir();
            }
            if(!check_file_file.exists()){
                boolean success = check_file_file.mkdirs();

            }
            try {

                deepFM = new DeepFM(MainActivity_travel.this);


            } catch (IOException e) {
                e.printStackTrace();
            }
            return responseString;
        }

        @Override
        protected void onPostExecute(String result) {
//            logArea.append("client ready \n");
            Message msg = new Message();
            Bundle bundle = new Bundle();
            bundle.putString("data", "success init");
            msg.setData(bundle);
            handler.sendMessage(msg);
            Log.d("INFO", "success init");
        }
    }
//
//    class RequestUpdateTask extends AsyncTask<Void, Integer, String> {
//        @Override
//        protected String doInBackground(Void... voids) {
//            String Train_time = "";
//            // requeest update
//            try {
//                int mVersion = response_json.getInt("model_version");
//                currentRound = response_json.getInt("current_round");
//                double testLoss = response_json.getDouble("model_loss");
//                double testAcc = response_json.getDouble("model_acc");
//                String upload_url = response_json.getString("upload_url");
//                String download_url = response_json.getString("model_url");
//                String Output_file_path = "/storage/self/primary/Download/save_model";
//                File check_file = new File(Output_file_path);
//
//                if(mVersion != model_version || mVersion == 0 || !ModelBuildCheck) {
//                    // model build
//                    cnn_model.buildModel(Output_file_path);
//                    ModelBuildCheck = true;
//                }
//
//                ui_testacc = testAcc;
//
//                Train_time = trainOneRound(currentRound, upload_url, mVersion);
//
//                // Memory Usage
//                final Runtime runtime = Runtime.getRuntime();
//                final long usedMemInMB=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
//                final long maxHeapSizeInMB=runtime.maxMemory() / 1048576L;
//                final long availHeapSizeInMB = maxHeapSizeInMB - usedMemInMB;
//
//                currentBatterytemp = Temperature();
//                StringBuffer cf = CpuFreq();
//
//                Request update_client_request = new Request.Builder()
//                        .url("http://" + address + ":8891/client_update?client_name=" + mUsername + "&current_round=" + Integer.toString(currentRound) + "&train_time=" + Train_time
//                                + "&heapsize=" + Long.toString(availHeapSizeInMB) + "&temperature=" + currentBatterytemp + "&cpu_freq=" + cf
//                                + "&acc_f1=" + acc_f1)
//                        .build();
//
//                response = client.newCall(update_client_request).execute();
//
//                if (response.isSuccessful()) {
//                    ResponseBody body = response.body();
//                    if (body != null) {
//                        responseString = body.string();
//                    }
//                }
//                else
//                    Log.d("INFO", "Connect Error Occurred");
//            } catch (JSONException | IOException e) {
//                e.printStackTrace();
//            } finally {
//                response.body().close();
//            }
//            return Train_time;
//        }
//
//        @Override
//        protected void onPostExecute(String result) {
//            Log.d("TRAINING TIME INFO", result);
//        }
//    }
//
//    class CheckModelVersion extends AsyncTask<Void, Integer, String> {
//        @Override
//        protected String doInBackground(Void... voids) {
//            Request update_client_request = new Request.Builder()
//                    .url("http://" + address + ":8891/model_version?version_client=" + mUsername + "&model_ver=" + model_version)
//                    .build();
//
//            try {
//                response = client.newCall(update_client_request).execute();
//
//                if (response.isSuccessful()) {
//                    ResponseBody body = response.body();
//                    if (body != null) {
//                        try {
//                            responseString = body.string();
//                            temp_modelstate = new JSONObject(responseString).getString("state");
//                        } catch (IOException | JSONException e) {
//                            e.printStackTrace();
//                        }
//                    }
//                } else
//                    Log.d("INFO", "Connect Error Occurred");
//            } catch (IOException e) {
//                e.printStackTrace();
//            } finally {
//                response.body().close();
//            }
//
//            return temp_modelstate;
//        }
//
//        @Override
//        protected void onPostExecute(String result) {
//            String modelVersion = "curent model version: " + result;
//            Log.d("INFO", modelVersion);
//        }
//    }
//
    private String trainOneRound(int currentRound, String upload_url, int modelVersion) throws IOException {
        Log.d("trainOneRound", "execute: train start!");
        long current_time = 0L;
        long train_time = 0L;
        try {
            current_time = System.currentTimeMillis();
            deepFM.train(4);
            train_time = System.currentTimeMillis();
            Log.d("TRAINING TIME INFO", Long.toString((train_time - current_time)/1000));
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Log.d("trainOneRound", "run: train finish!");

        // save trained model
        String AndroidModelPath = "/storage/self/primary/Download/save_weight/";

        deepFM.saveSerializeModel("weight_" + mUsername + ".json");

        Log.d("MODEL INFO", "Complete model save!");
        // upload to server trained model
//        deepFM.uploadTo(AndroidModelPath + "weight_" + mUsername + ".json", upload_url, client);

        return Long.toString((train_time - current_time)/1000);
    }

}
